// Garion Cox
// Dec 13, 2021
// CS 1410 Final Project

using System;

namespace NeuralNetwork.Main
{
    class Program
    {
        static void Main()
        {
            // Input layer
            InputNode inputNode1 = new InputNode();
            InputNode inputNode2 = new InputNode();
            
            // Hidden Layer
            AddNode addNode1 = new AddNode();

            // Cost node
            CostNode outputNode = new CostNode();

            // Scalar
            double gamma = 0.1;

            // Bias
            ValueNode b = new ValueNode(0);

            // Arrange network
            Console.WriteLine("Arranging Network... ");

            addNode1.linkInputNode(inputNode1);
            addNode1.linkInputNode(inputNode2);
            addNode1.linkInputNode(b);
            outputNode.linkInputNode(addNode1);

            Console.WriteLine("Arranged!");

            // Set values
            Console.Write("\nSetting input values: {3, 5}... ");
            inputNode1.Value = 3;
            inputNode2.Value = 5;
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("Values set!");
            Console.ResetColor();

            Console.Write("Setting expected value: 10...");
            outputNode.ActualValue = 10;
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("Values set!");
            Console.ResetColor();

            Console.Write("(Press Any Key To Continue)");
            Console.ReadLine();

            // Train the network
            Console.WriteLine("\n\nOverfitting the network with one training example:");

            while (true) {
                Console.Write($"Expected Value (10), actual value: ");
                double prediction = addNode1.getOutput();

                if ( prediction >= 9.9 && prediction <= 10.1 ) {
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine($"({prediction})");
                    Console.ResetColor();
                    break;
                }

                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"({prediction})");
                Console.ResetColor();

                double scaleValue = b.getAdjoint()*gamma;
                Console.Write("Adjusted bias by ");
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"({-scaleValue})");
                Console.ResetColor();
                b.Value -= scaleValue;
            }
        }
    }
}
