// Garion Cox
// Dec 13, 2021
// CS 1410 Final Project

using System.Collections.Generic;

namespace NeuralNetwork
{
    // REQUIREMENT #14: Second Interface
    public interface INetwork
    {
        List<double> GetPrediction(List<double> inputValues);
        void Train(DataPoint<double> dataValue);
        void updateActualOutputValues(List<double> values);
        void updateInputValues(List<double> values);
    }
}