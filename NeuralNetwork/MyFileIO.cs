// Garion Cox
// Dec 13, 2021
// CS 1410 Final Project

using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{

    // Modified From: https://jamesmccaffrey.wordpress.com/2013/11/23/reading-the-mnist-data-set-with-c/
    // James D. McCaffrey, Nov 23, 2013. Accessed Nov 30, 2021.
    public class FileIO
    {

        // REQUIREMENT #12: Static Member Function
        // EXTRA CREDIT #1: Working With File IO
        public static List<DataPoint<byte>> LoadData(string labelsPath, string imagesPath) {
            Console.WriteLine("\nBegin Data Loading:");
            FileStream ifsLabels = new FileStream(labelsPath, FileMode.Open); // test labels
            FileStream ifsImages = new FileStream(imagesPath, FileMode.Open); // test images

            BinaryReader brLabels = new BinaryReader(ifsLabels);
            BinaryReader brImages = new BinaryReader(ifsImages);
    
            int magic1 = brImages.ReadInt32(); // discard
            int numImages = brImages.ReadInt32(); 
            int numRows = brImages.ReadInt32(); 
            int numCols = brImages.ReadInt32(); 

            int magic2 = brLabels.ReadInt32(); 
            int numLabels = brLabels.ReadInt32(); 

            List<byte> pixels = new List<byte>();

            // each test image
            List<DataPoint<byte>> digits = new List<DataPoint<byte>>();
            for (int di = 0; di < 10000; ++di) 
            {
                for (int i = 0; i < 28; ++i) 
                {
                    for (int j = 0; j < 28; ++j) 
                    {
                        byte b = brImages.ReadByte();
                        pixels.Add(b);
                    }
                }

                // get label in format [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                List<byte> label = new List<byte>();
                byte actualValue = brLabels.ReadByte();
                for (int i = 0; i < 10; i++) {
                    if (i == actualValue) {
                        label.Add(1);
                    } else {
                        label.Add(0);
                    }
                }

                digits.Add(new DataPoint<byte>(pixels, label));
                pixels = new List<byte>();
            }

            ifsImages.Close();
            brImages.Close();
            ifsLabels.Close();
            brLabels.Close();

            Console.WriteLine($"Loaded {digits.Count} images. End\n");
            return digits;
        }

        // Convert the DataPoint<byte> object into a DataPoint<double> object
        public static DataPoint<double> ConvertToDouble(DataPoint<byte> d) {
            List<double> byteInputData = new List<double>(d.InputData.Select(b => Convert.ToDouble(b)));
            List<double> byteActualValues = new List<double>(d.ActualValues.Select(b => Convert.ToDouble(b)));

            return new DataPoint<double>(byteInputData, byteActualValues);
        }
    }
}