using Microsoft.ML.Data;
using Microsoft.ML;
using SeriesTemporales.Model;
using System.Data;
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
        SsaForecastingTransformer forecasterTransformer = forecastingPipeline.Fit(dataView);
        Evaluate(testingDataView, forecasterTransformer, mlContext);
        //var forecastEngine = forecasterTransformer.CreateTimeSeriesEngine<InputModel, OutputModel>(mlContext);  //<- System.MissingMethodException: 
        //'Method not found: 'Void Microsoft.ML.PredictionEngineBase`2..ctor(Microsoft.ML.Runtime.IHostEnvironment, Microsoft.ML.ITransformer, Boolean, Microsoft.ML.Data.SchemaDefinition, Microsoft.ML.Data.SchemaDefinition)'.'
        Forecast(testingDataView, 6, forecastEngine, mlContext);
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
        // Método predict
#pragma warning disable CS8600 // Se va a convertir un literal nulo o un posible valor nulo en un tipo que no acepta valores NULL
        var predictMethod = typeof(TimeSeriesPredictionEngine<InputModel, OutputModel>).GetMethod("Predict", new Type[] { typeof(Int32), typeof(Single) });
        var forecast = (OutputModel)predictMethod.Invoke(engine, new object[] { 6, 0.95f });
#pragma warning restore CS8600 // Se va a convertir un literal nulo o un posible valor nulo en un tipo que no acepta valores NULL
        IEnumerable<string> forecastOutput = mlContext.Data.CreateEnumerable<InputModel>(testData, reuseRowObject: false)
        .Take(horizon)
        .Select((InputModel rental, int index) =>
        {
            string date = rental.Month.ToString(@"yyyy-MM");
            float actualPassengers = rental.Passengers;
            float lowerEstimate = Math.Max(0, forecast.LowerBoundPassengers[index]);
            float estimate = forecast.ForecastPassengers[index];
            float upperEstimate = forecast.UpperBoundPassengers[index];
            return $"Fecha: {date}\n" +
            $"Pasajeros reales: {actualPassengers}\n" +
            $"Lower Estimate: {lowerEstimate}\n" +
            $"Prediccion: {estimate}\n" +
            $"Upper Estimate: {upperEstimate}\n";
        });
        Console.WriteLine(" > Passengers Forecast");
        Console.WriteLine("");
        foreach (var prediction in forecastOutput)
        {
            Console.WriteLine(prediction);
        }
    }
}