// Garion Cox
// Dec 13, 2021
// CS 1410 Final Project

namespace NeuralNetwork
{

    // REQUIREMENT #6: Inheritance
    public class AddNode : AbstractOperationNode
    {
        public AddNode() : base() { }

        // Add all the inputs together
        public override void doOperation() {
            _output = 0;
            foreach (AbstractNode n in _inputNodes) {
                _output += n.getOutput();
            }
        }

        // the derivative will always be 1, regardless of the input
        public override double doDerivativeOperation(AbstractNode n) {
            return 1;
        }
    }
}