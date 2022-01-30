// Garion Cox
// Dec 13, 2021
// CS 1410 Final Project

namespace NeuralNetwork
{
    // An abstract constant node that:
    // 1) Contains a value getter and setter
    // 2) Can return its output

    // REQUIREMENT #3: Third Class Definition
    public abstract class AbstractConstantNode : AbstractNode
    {
        public double Value { get; set; }
        public AbstractConstantNode(double value) {
            Value = value;
        }

        public override double getOutput() {
            HasComputedOutput = true;
            HasComputedAdjoint = false;
            return Value;
        }
    }
}