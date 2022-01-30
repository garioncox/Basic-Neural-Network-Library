// Garion Cox
// Dec 13, 2021
// CS 1410 Final Project

using System.Collections.Generic;

namespace NeuralNetwork
{
    // A generic datapoint class that has:
    // 1) A list of data
    // 2) A list of expected values

    // REQUIREMENT #10: Generic Datatype
    public class DataPoint<T> {
        public List<T> InputData { get; protected set; }
        public List<T> ActualValues { get; protected set; }

        public DataPoint(List<T> data, List<T> values) {
            InputData = data;
            ActualValues = values;
        }
    }
}   