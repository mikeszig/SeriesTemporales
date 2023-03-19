namespace SeriesTemporales.Model
{
    public class OutputModel
    {
        public float[] ForecastPassengers { get; set; }
        public float[] LowerBoundPassengers { get; set; }
        public float[] UpperBoundPassengers { get; set; }
    }
}