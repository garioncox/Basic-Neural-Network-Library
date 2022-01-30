// Garion Cox
// Dec 13, 2021
// CS 1410 Final Project

namespace NeuralNetwork
{
    // REQUIREMENT #13: Interface
    public interface INode
    {
        bool HasComputedOutput { get; }
        bool HasComputedAdjoint { get; }
        double getOutput();
        double getAdjoint();
    }
}