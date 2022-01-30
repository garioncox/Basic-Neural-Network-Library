// Garion Cox
// Dec 13, 2021
// CS 1410 Final Project

using System;

namespace NeuralNetwork
{
    public class SigmoidNode : AbstractOperationNode
    {
        public SigmoidNode() : base() { }
        public override void doOperation() {
            _output = 0;
            foreach (AbstractNode n in _inputNodes) {
                _output += n.getOutput();
            }

            // Sigmoid: 1 / (1 + e^-x)
            _output = sigmoid(_output);
        }

        // derivative: sigmoid(x) * (1 - sigmoid(x))
        public override double doDerivativeOperation(AbstractNode n) {
            return sigmoid(n.getOutput()) * (1 - sigmoid(n.getOutput()) );
        }

        double sigmoid (double x) {
            return 1 / (1 + Math.Exp(-1*x));
        }
    }
}