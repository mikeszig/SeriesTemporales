using Microsoft.ML.Data;
using Microsoft.ML;
using SeriesTemporales.Model;
using MySql.Data.MySqlClient;
using System.Data.SqlClient;
using System.Data;
using Microsoft.ML.Transforms.TimeSeries;
using System.Data.Common;
using MySqlConnector;

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
        //var dataViewSql = LoadFromDatabase(connectionString, sqlCommand, mlContext);
        //var result = dataViewSql.Preview(); // <- Extrae correctamente los datos de la base pero la columna Month no la parsea - salta excepción constantemente
        IDataView dataView = mlContext.Data.LoadFromTextFile<InputModel>
                                    (
                                    Path.Combine(Environment.CurrentDirectory, "Data", "AirPassengers.csv"),
                                    hasHeader: true,
                                    separatorChar: ','
                                    );
        //var result = dataView.Preview();
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
        SsaForecastingTransformer forecasterTransformer = forecastingPipeline.Fit(dataView);
        Evaluate(testingDataView, forecasterTransformer, mlContext);
        var forecastEngine = forecasterTransformer.CreateTimeSeriesEngine<InputModel, OutputModel>(mlContext); //<- System.MissingMethodException: 
        //'Method not found: 'Void Microsoft.ML.PredictionEngineBase`2..ctor(Microsoft.ML.Runtime.IHostEnvironment, Microsoft.ML.ITransformer, Boolean, Microsoft.ML.Data.SchemaDefinition, Microsoft.ML.Data.SchemaDefinition)'.'
        Forecast(testingDataView, 6, forecastEngine, mlContext);
    }

    private static IDataView LoadFromDatabase(string connectionString, string sqlCommand, MLContext mlContext)
    {
        DatabaseLoader loader = mlContext.Data.CreateDatabaseLoader<InputModel>();
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