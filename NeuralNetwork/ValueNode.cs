// Garion Cox
// Dec 13, 2021
// CS 1410 Final Project

namespace NeuralNetwork
{
    public class ValueNode : AbstractConstantNode
    {
        public ValueNode(double value) : base(value) { }

        public void scaleValue(double gamma) {
            Value -= getAdjoint() * gamma;
        }
    }
}