// Garion Cox
// Dec 13, 2021
// CS 1410 Final Project

using System.Collections.Generic;

namespace NeuralNetwork
{
    // An Abstract Operation Node that can:
    // 1) Recursively compute its output
    // 2) Link with an input node

    // REQUIREMENT #2: Second Class Definition
    public abstract class AbstractOperationNode : AbstractNode
    {
        protected List<AbstractNode> _inputNodes = new List<AbstractNode>();
        public abstract void doOperation();
        public abstract double doDerivativeOperation(AbstractNode n);

        public AbstractOperationNode() { }

        // Adds an input connection to a node. Only this node knows it's connected to the other node
        public void addInputNode(AbstractNode node) {
            _inputNodes.Add(node);
        }

        public override double getOutput() {
            if (!HasComputedOutput) {

                // Make sure all the input nodes have computed their output
                foreach (AbstractNode n in _inputNodes) {
                    n.getOutput();
                }

                // Do the specified operation on the inputs
                doOperation();
                HasComputedOutput = true;
                HasComputedAdjoint = false;
            }

            return _output;
        }

        // Link the two nodes together. Both nodes know that the other one exists
        public void linkInputNode(AbstractNode n) {
            _inputNodes.Add(n);
            n.addOutputNode(this);
        }
    }
}