using Microsoft.ML.Data;

namespace SeriesTemporales.Model
{
    public class InputModel
    {
        //[System.ComponentModel.DataAnnotations.DisplayFormat(ApplyFormatInEditMode = true, DataFormatString = "{0:yyyy-MM}")]
        public string Month { get; set; }
        public Single Passengers { get; set; }
    }
}

