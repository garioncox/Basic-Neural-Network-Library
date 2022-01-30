// Garion Cox
// Dec 13, 2021
// CS 1410 Final Project

using System;

namespace NeuralNetwork
{
    // REQUIREMENT #7: Polymorphism
    public class CostNode : AbstractOperationNode
    {
        // The value that is expected
        public double ActualValue { get; set; }
        double _sum;
        public CostNode() : base() { }

        // Output is ((sum of previous nodes) - y)^2
        public override void doOperation() {
            _output = 0;
            _sum = 0;
            foreach (AbstractNode n in _inputNodes) {
                _sum += n.getOutput();
            }

            _output = Math.Pow( (_sum - ActualValue), 2);
        }

        // derivative is 2*((sum of previous nodes) - y)
        public override double doDerivativeOperation(AbstractNode n) {
            return 2*(_sum - ActualValue);
        }

        public double getSum() {
            if (!HasComputedOutput) {
                getOutput();
            }
            return _sum;
        }
    }
}