// Garion Cox
// Dec 13, 2021
// CS 1410 Final Project

namespace NeuralNetwork
{
    // REQUIREMENT #8: Second Example of Polymorphism
    public class MultiplyNode : AbstractOperationNode
    {
        public MultiplyNode() : base() { }

        // Multiply the inputs together
        public override void doOperation() {
            _output = 1;
            foreach (AbstractNode n in _inputNodes) {
                _output *= n.getOutput();
            }
        }

        // The derviative is this node's output / the passed in node's output.
        // the partial derivative of 'abc' with respect to 'a' = bc
        public override double doDerivativeOperation(AbstractNode n) {
            return _output / n.getOutput();
        }
    }
}