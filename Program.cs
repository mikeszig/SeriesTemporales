using Microsoft.ML.Data;
using Microsoft.ML;
using SeriesTemporales.Model;
using MySql.Data.MySqlClient;
using System.Data.SqlClient;
using System.Data;
using Microsoft.ML.Transforms.TimeSeries;
using System.Data.Common;
using MySqlConnector;
using NUnit.Framework.Internal;
using System.Reflection;
using Microsoft.ML.Runtime;

public class Program
{
    //páginas de interés:
    //https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/time-series-demand-forecasting
    //https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.timeseriescatalog.forecastbyssa?view=ml-dotnet

    private static string connectionString = @"server=db4free.net;user=elchinobd;database=elchinobd;password=elchinobd;";
    private static string sqlCommand = "SELECT * FROM airpassengers";

    public static void Main(string[] args)
    {
        MLContext mlContext = new(seed: 0);
        var data = LoadFromDatabase(connectionString, sqlCommand, mlContext);
        IDataView dataView = FormatIDataView(data, mlContext);
        var result = dataView.Preview(); // <- poner punto de ruptura para revisar el contenido del dataView
        var splitter = mlContext.Data.TrainTestSplit(dataView, 0.2);
        IDataView trainingDataView = splitter.TrainSet;
        IDataView testingDataView = splitter.TestSet;
        var forecastingPipeline = mlContext.Forecasting.ForecastBySsa
                                    (
                                    inputColumnName: nameof(InputModel.Passengers),
                                    outputColumnName: nameof(OutputModel.ForecastPassengers),
                                    confidenceLowerBoundColumn: nameof(OutputModel.LowerBoundPassengers),
                                    confidenceUpperBoundColumn: nameof(OutputModel.UpperBoundPassengers),
                                    trainSize: 114,
                                    seriesLength: 12,
                                    windowSize: 6,
                                    horizon: 6,
                                    confidenceLevel: 0.95f
                                    );
        ITransformer/*Antes: SsaForecastingTransformer*/ forecasterTransformer = forecastingPipeline.Fit(trainingDataView);
        Evaluate(testingDataView, forecasterTransformer, mlContext);
        //var forecastEngine = forecasterTransformer.CreateTimeSeriesEngine<InputModel, OutputModel>(mlContext);  //<- System.MissingMethodException: 
        //'Method not found: 'Void Microsoft.ML.PredictionEngineBase`2..ctor(Microsoft.ML.Runtime.IHostEnvironment, Microsoft.ML.ITransformer, Boolean, Microsoft.ML.Data.SchemaDefinition, Microsoft.ML.Data.SchemaDefinition)'.'
        //Forecast(testingDataView, 6, forecastEngine, mlContext);

        // Constructor TimeSeriesPredictionEngine<InputModel, OutputModel>
        var constructorMethod = typeof(PredictionFunctionExtensions)
                                    .GetMethod(nameof(PredictionFunctionExtensions.CreateTimeSeriesEngine),
                                    new Type[] { 
                                        typeof(ITransformer), 
                                        typeof(IHostEnvironment), 
                                        typeof(Boolean),
                                        typeof(SchemaDefinition),
                                        typeof(SchemaDefinition)});
        var constructor = constructorMethod.MakeGenericMethod(new Type[] { typeof(InputModel), typeof(OutputModel) });
        var engine = constructor.Invoke(forecasterTransformer, 
            new object[] { forecasterTransformer, mlContext, true, 
                SchemaDefinition.Create(typeof(InputModel)), SchemaDefinition.Create(typeof(OutputModel)) }); // <- [SOLVED]Parameter count mismatch / Method not found

        // Método predict
        var predictMethod = typeof(TimeSeriesPredictionEngine<InputModel, OutputModel>).GetMethod("Predict", new Type[] { typeof(Int32), typeof(Single) });
        var output = predictMethod.Invoke(engine, new object[] { 6, 0.95f });
    }

    private static IDataView FormatIDataView(IDataView idv, MLContext mlContext)
    {
        List<InputModel> inputs = new List<InputModel>();
        int iRow = 0;
        var FEATURE = Array.ConvertAll(idv.GetColumn<UInt32>("Passengers").ToArray(), item => Convert.ToSingle(item));
        var TARGET = idv.GetColumn<string>("Month").ToArray();
        foreach (var row in FEATURE)
        {
            //object columnValue = Convert.ToInt32(row); <- OLD VERSION
            var obs = new InputModel();
            obs.Month = TARGET[iRow];
            obs.Passengers = row;
            inputs.Add(obs);
        }
        IEnumerable<InputModel> inputsIEnum = inputs;
        var definedSchema = SchemaDefinition.Create(typeof(InputModel));
        IDataView trainingDataView = mlContext.Data.LoadFromEnumerable(inputsIEnum, definedSchema);
        return trainingDataView;
    }

    private static IDataView LoadFromDatabase(string connectionString, string sqlCommand, MLContext mlContext)
    {
        DatabaseLoader loader = mlContext.Data.CreateDatabaseLoader<DataModel>();
        DatabaseSource dbSource = new DatabaseSource(MySqlConnectorFactory.Instance, connectionString, sqlCommand);
        IDataView data = loader.Load(dbSource);
        return data;
    }

    private static void Evaluate(IDataView testData, ITransformer model, MLContext mlContext)
    {
        IDataView predictions = model.Transform(testData);
        IEnumerable<float> actual = mlContext.Data.CreateEnumerable<InputModel>(testData, true).Select(observed => observed.Passengers);
        IEnumerable<float> forecast = mlContext.Data.CreateEnumerable<OutputModel>(predictions, true).Select(prediction => prediction.ForecastPassengers[0]);
        var metrics = actual.Zip(forecast, (actualValue, forecastValue) => actualValue - forecastValue);
        var MAE = metrics.Average(error => Math.Abs(error));
        var RMSE = Math.Sqrt(metrics.Average(error => Math.Pow(error, 2)));
        Console.WriteLine(" > Evaluation Metrics");
        Console.WriteLine($" > Mean Absolute Error: {MAE:F3}");
        Console.WriteLine($" > Root Mean Squared Error: {RMSE:F3}\n");
    }

    private static void Forecast(IDataView testData, int horizon, TimeSeriesPredictionEngine<InputModel, OutputModel> engine, MLContext mlContext)
    {
        OutputModel forecast = engine.Predict();
        IEnumerable<string> forecastOutput = mlContext.Data.CreateEnumerable<InputModel>(testData, reuseRowObject: false)
        .Take(horizon)
        .Select((InputModel rental, int index) =>
        {
            string rentalDate = rental.Month;
            float actualRentals = rental.Passengers;
            float lowerEstimate = Math.Max(0, forecast.LowerBoundPassengers[index]);
            float estimate = forecast.ForecastPassengers[index];
            float upperEstimate = forecast.UpperBoundPassengers[index];
            return $"Date: {rentalDate}\n" +
            $"Actual Rentals: {actualRentals}\n" +
            $"Lower Estimate: {lowerEstimate}\n" +
            $"Forecast: {estimate}\n" +
            $"Upper Estimate: {upperEstimate}\n";
        });
        Console.WriteLine(" > Rental Forecast");
        Console.WriteLine("");
        foreach (var prediction in forecastOutput)
        {
            Console.WriteLine(prediction);
        }
    }
}