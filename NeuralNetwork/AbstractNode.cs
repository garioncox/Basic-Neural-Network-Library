// Garion Cox
// Dec 13, 2021
// CS 1410 Final Project

using System.Collections.Generic;

namespace NeuralNetwork
{

    // An abstract node that has:
    // 1) A list of outputs
    // 2) Knows if it's computed its output and adjoint
    // 3) A cached output and adjoint value
    // 4) A way to recursively compute its adjoint

    // REQUIREMENT #1: Class Definition
    public abstract class AbstractNode : INode
    {
        protected List<AbstractNode> _outputNodes = new List<AbstractNode>();

        // REQUIREMENT #11: Properties
        public bool HasComputedOutput { get; set; }
        public bool HasComputedAdjoint { get; set; }
        double _adjoint;
        protected double _output;

        public void addOutputNode(AbstractOperationNode n) {
            _outputNodes.Add(n);
        }
        
        public double getAdjoint() {

            if (!HasComputedAdjoint) {

                if (_outputNodes.Count == 0) {
                    _adjoint = 1;
                }
                
                // Get the adjoints from all the outputs, then return sum( derivative with respect to this * output.Adjoint )
                else {
                    _adjoint = 0;
                    foreach (AbstractOperationNode n in _outputNodes) {

                        // EXTRA CREDIT #2: Recursion
                        _adjoint += n.doDerivativeOperation(this)*n.getAdjoint();
                    }
                }
            }
            
            HasComputedAdjoint = true;
            HasComputedOutput = false;

            return _adjoint;
        }

        public abstract double getOutput();
    }
}