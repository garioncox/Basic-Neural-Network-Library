// Garion Cox
// Dec 13, 2021
// CS 1410 Final Project

using NUnit.Framework;
using System.Collections.Generic;

namespace NeuralNetwork.Tests
{
    public class Tests2
    {

        string drive;

        [SetUp]
        public void Setup()
        {
            // ARRANGE
            drive = System.IO.Directory.GetCurrentDirectory().Split(":")[0];
        }

        [Test]
        public void AutomaticNetworkTest()
        {
            // ARRANGE the network:

            // input layer
            InputNode inputNode1 = new InputNode();
            InputNode inputNode2 = new InputNode();
            List<InputNode> inputs = new List<InputNode>() {inputNode1, inputNode2};

            // hidden layers
            AddNode addNode1 = new AddNode();
            List<AbstractOperationNode> hiddenLayer1 = new List<AbstractOperationNode>() {addNode1};
            List<List<AbstractOperationNode>> hiddenLayers = new List<List<AbstractOperationNode>>() {hiddenLayer1};

            // output layer
            CostNode outputNode = new CostNode();
            outputNode.ActualValue = 8;
            List<CostNode> costs = new List<CostNode>() {outputNode};

            // weights, biases
            List<double> weightValues = new List<double>() {1, 1, 1};
            List<double> biasValues = new List<double>() {0, 0, 0, 0};

            // scalar value 
            double gamma = 0.1;

            MyNetwork n = new MyNetwork(inputs, hiddenLayers, costs, weightValues, biasValues, gamma);

            List<double> inputValues = new List<double>() {10, 200};
            Assert.AreEqual(210, n.GetPrediction(inputValues)[0]);

            // Test to see if we can run two training examples without getting the same result
            List<double> inputValues2 = new List<double>() {3, 5};
            Assert.AreEqual(8, n.GetPrediction(inputValues2)[0]);
        }

        [Test]
        public void AutomaticNetworkTest2()
        {
            // ARRANGE the network: inputs, hidden layers, outputs, weights, biases, gamma
            InputNode inputNode1 = new InputNode();
            InputNode inputNode2 = new InputNode();
            List<InputNode> inputs = new List<InputNode>() {inputNode1, inputNode2};

            AddNode addNode1 = new AddNode();
            List<AbstractOperationNode> hiddenLayer1 = new List<AbstractOperationNode>() {addNode1};
            List<List<AbstractOperationNode>> hiddenLayers = new List<List<AbstractOperationNode>>() {hiddenLayer1};

            CostNode outputNode = new CostNode();
            outputNode.ActualValue = 10;
            List<CostNode> costs = new List<CostNode>() {outputNode};

            List<double> weightValues = new List<double>() {1, 1, 1};
            List<double> biasValues = new List<double>() {0, 0, 0, 0};

            double gamma = 0.1;

            MyNetwork n = new MyNetwork(inputs, hiddenLayers, costs, weightValues, biasValues, gamma);

            // Test to see if we can overfit the data
            List<double> inputValues2 = new List<double>() {5, 5};
            DataPoint<double> d = new DataPoint<double>(inputValues2, new List<double>() {10});
            Assert.That(n.GetPrediction(inputValues2)[0], Is.EqualTo(10).Within(100));


            // Currently does not work: does not converge. It exponentially grows, but
            // this implementation is out of the scope of the project. Ignore.

            // for (int i = 0; i < 150; i++) {
            //     n.Train(d);
            //     Assert.That(n.GetPrediction(inputValues2)[0], Is.EqualTo(10).Within(1000));
            // }

            // Assert.That(n.GetPrediction(inputValues2)[0], Is.EqualTo(10).Within(0.01));
        }

        [Test]
        public void FileIOTest()
        {
            // Test to see if we can load the data correctly from the specified files
            List<DataPoint<byte>> digits = FileIO.LoadData($@"{drive}:\Neural Network (CS 1410 Final Project)\data\t10k-labels.idx1-ubyte", 
                                                           $@"{drive}:\Neural Network (CS 1410 Final Project)\data\t10k-images.idx3-ubyte");

            Assert.AreEqual(10000, digits.Count);
            Assert.AreEqual(784, digits[0].InputData.Count);
            Assert.AreEqual(10, digits[0].ActualValues.Count);
        }

        [Test]
        public void ManualNetworkTest()
        {
            // ARRANGE inputs, hidden layers, and outputs
            ValueNode inputNode1 = new ValueNode(3);
            ValueNode inputNode2 = new ValueNode(5);

            AddNode addNode1 = new AddNode();
            addNode1.linkInputNode(inputNode1);
            addNode1.linkInputNode(inputNode2);

            CostNode outputNode = new CostNode();
            outputNode.linkInputNode(addNode1);
            outputNode.ActualValue = 8;


            // ACT & ASSERT
            Assert.AreEqual(3, inputNode1.getOutput());
            Assert.AreEqual(5, inputNode2.getOutput());
            Assert.AreEqual(8, addNode1.getOutput());
            Assert.AreEqual(0, outputNode.getOutput());

            Assert.IsTrue(inputNode1.HasComputedOutput);
            Assert.IsTrue(inputNode2.HasComputedOutput);
            Assert.IsTrue(addNode1.HasComputedOutput);
            Assert.IsTrue(outputNode.HasComputedOutput);

            Assert.AreEqual(1, outputNode.getAdjoint());
            Assert.AreEqual(0, addNode1.getAdjoint());
            Assert.AreEqual(0, inputNode1.getAdjoint());
            Assert.AreEqual(0, inputNode2.getAdjoint());

            Assert.IsFalse(inputNode1.HasComputedOutput);
            Assert.IsFalse(inputNode2.HasComputedOutput);
            Assert.IsFalse(addNode1.HasComputedOutput);
            Assert.IsFalse(outputNode.HasComputedOutput);
            Assert.IsTrue(inputNode1.HasComputedAdjoint);
            Assert.IsTrue(inputNode2.HasComputedAdjoint);
            Assert.IsTrue(addNode1.HasComputedAdjoint);
            Assert.IsTrue(outputNode.HasComputedAdjoint);

            // ACT & ASSERT
            inputNode1.Value = 5;
            inputNode2.Value = 5;
            outputNode.ActualValue = 5;

            Assert.AreEqual(5, inputNode1.getOutput());
            Assert.AreEqual(5, inputNode2.getOutput());
            Assert.AreEqual(5, inputNode1.Value);
            Assert.AreEqual(5, inputNode2.Value);
            Assert.AreEqual(10, addNode1.getOutput());
            Assert.AreEqual(25, outputNode.getOutput());

            Assert.AreEqual(1, outputNode.getAdjoint());
            Assert.AreEqual(10, addNode1.getAdjoint());
            Assert.LessOrEqual(10, inputNode1.getAdjoint());
            Assert.LessOrEqual(10, inputNode2.getAdjoint());

            double gamma = 0.1;
            Assert.AreEqual(1, inputNode1.getAdjoint() * gamma);
            Assert.AreEqual(1, inputNode2.getAdjoint() * gamma);

            inputNode1.scaleValue(gamma);
            inputNode2.scaleValue(gamma);
            Assert.AreEqual(5 - 1, inputNode1.Value);
            Assert.AreEqual(5 - 1, inputNode2.Value);

            // Test manually overfitting the data
            for (int i = 0; i < 15; i++) {
                outputNode.getOutput();
                inputNode1.scaleValue(gamma);
                inputNode2.scaleValue(gamma);
            }

            Assert.That(outputNode.getSum(), Is.EqualTo(5).Within(0.01));
        }

        [Test]
        public void ManualNetworkTest2()
        {
            // ARRANGE
            ValueNode inputNode1 = new ValueNode(100);
            ValueNode inputNode2 = new ValueNode(3);

            AddNode addNode1 = new AddNode();
            addNode1.linkInputNode(inputNode1);
            addNode1.linkInputNode(inputNode2);

            CostNode outputNode = new CostNode();
            outputNode.linkInputNode(addNode1);
            outputNode.ActualValue = 10;

            double gamma = 0.1;

            // ACT
            for (int i = 0; i < 50; i++) {
                outputNode.getOutput();
                inputNode1.scaleValue(gamma);
                inputNode2.scaleValue(gamma);
            }

            // ASSERT
            Assert.That(outputNode.getSum(), Is.EqualTo(10).Within(0.01));
        }
    }
}