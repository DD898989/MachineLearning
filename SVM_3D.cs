using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using System.Linq;
using Data;
using NeuralNetwork;
using Plot3D;

namespace WindowsFormsApplication23
{
    public partial class Form1 : Form
    {
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        static extern bool AllocConsole();

        private ScatterPlot _scatterPlot1;

        public static void Swap<T>(IList<T> list, int indexA, int indexB)
        {
            T tmp = list[indexA];
            list[indexA] = list[indexB];
            list[indexB] = tmp;
        }
        // ----------------------------------------------------------------------------------------
        private void Draw()
        {
            Random rand = new Random();
            List<double[]> Points = new List<double[]>();
            double[][] /*    */DATA_ori = new data().data2;
            List<List<double>> DATA_temp = new List<List<double>>();
            int _sr = 50;//seperate plain

            DATA_temp = DATA_ori
            .Where(inner => inner != null) // Cope with uninitialised inner arrays.
            .Select(inner => inner.ToList()) // Project each inner array to a List<string>
            .ToList(); // Materialise the IEnumerable<List<string>> to List<List<string>>

            if (false)  //if data feature more than 3D space, remove some
                foreach (List<double> db in DATA_temp)
                    db.RemoveAt(2);

            List<List<double>> c1 = new List<List<double>>();
            List<List<double>> c2 = new List<List<double>>();
            List<List<double>> c3 = new List<List<double>>();

            foreach (List<double> _list in DATA_temp)
            {
                if (_list[3] == 1)
                    c1.Add(_list);
                else if (_list[4] == 1)
                    c2.Add(_list);
                else if (_list[5] == 1)
                    c3.Add(_list);
            }

            int minCount = new[] { c1.Count, c2.Count, c3.Count }.Min();
            DATA_temp.Clear();

            Random rnd = new Random();
            while (c1.Count > minCount)
                c1.RemoveAt(rnd.Next(c1.Count));
            while (c2.Count > minCount)
                c2.RemoveAt(rnd.Next(c2.Count));
            while (c3.Count > minCount)
                c3.RemoveAt(rnd.Next(c3.Count));

            DATA_temp = c1.Concat(c2).Concat(c3).ToList();

            double[][] DATA_dum = new double[DATA_temp.Count][];
            for (int i = 0; i < DATA_temp.Count; i++)
                DATA_dum[i] = new double[] { 
                    DATA_temp[i][0],
                    DATA_temp[i][1],
                    DATA_temp[i][2],
                    DATA_temp[i][3],
                    DATA_temp[i][4],
                    DATA_temp[i][5]
                };
            double[][] DATA_fail = new double[DATA_dum.Length][];
            int failCount = 0;
            P.TrainAndTestTrainData(DATA_dum, DATA_fail, 3, 6, 3, ref failCount);
            //-----------------------------------------------------------------------------------------------------------
            //draw seperate plain

            double[] max3 = new double[] { double.MinValue, double.MinValue, double.MinValue };
            double[] min3 = new double[] { double.MaxValue, double.MaxValue, double.MaxValue };
            for (int i = 0; i < DATA_dum.Length; i++)
                for (int j = 0; j < DATA_dum[0].Length - 3/*numInput*/; j++)
                {
                    if (DATA_dum[i][j] > max3[j])
                        max3[j] = DATA_dum[i][j];

                    if (DATA_dum[i][j] < min3[j])
                        min3[j] = DATA_dum[i][j];
                }

            //_sampleData = new double[30][][][];
            //_sampleData[i] = new double[30][][];
            //_sampleData[i][j] = new double[30][];
            //_sampleData[i][j][k] = new double[] { x, y, z };
            double[][][][] _sampleData = new double[_sr][][][];
            for (int i = 0; i < _sr; i++)
            {
                _sampleData[i] = new double[_sr][][];
                for (int j = 0; j < _sr; j++)
                {
                    _sampleData[i][j] = new double[_sr][];
                    for (int k = 0; k < _sr; k++)
                        _sampleData[i][j][k] = new double[] {  
                            min3[0]+i*((max3[0]-min3[0])/(_sr-1)),   
                            min3[1]+j*((max3[1]-min3[1])/(_sr-1)),  
                            min3[2]+k*((max3[2]-min3[2])/(_sr-1))};
                }
            }

            int[][] sur = { //surrounding samples
                            new int[] { +1, 0, 0 } , new int[] { -1, 0, 0 } , new int[] { 0, +1, 0 } ,
                            new int[] { 0, -1, 0 } , new int[] { 0, 0, +1 } , new int[] { 0, 0, -1 } ,
                        };

            for (int i = 1; i < _sr - 1; i++)
                for (int j = 1; j < _sr - 1; j++)
                    for (int k = 1; k < _sr - 1; k += 2) //+=2 for sample point that not next to each other
                    {
                        if (k == 1)
                        {
                            //for sample point that not next to each other in 3D space

                            if (j % 2 == 0 && i % 2 != 0)
                                k += 1;
                            if (j % 2 != 0 && i % 2 == 0)
                                k += 1;
                        }

                        double[] currentSample = new double[] { 
                            _sampleData[i][j][k][0],//x
                            _sampleData[i][j][k][1],//y
                            _sampleData[i][j][k][2] //z
                        };

                        int classCurrent = P.Classify(currentSample);

                        for (int m = 0; m < sur.Length; m++)
                        {
                            int classSurrounding = P.Classify(
                                _sampleData[i + sur[m][0]][j + sur[m][1]][k + sur[m][2]][0], //x +- 1or0
                                _sampleData[i + sur[m][0]][j + sur[m][1]][k + sur[m][2]][1], //y +- 1or0
                                _sampleData[i + sur[m][0]][j + sur[m][1]][k + sur[m][2]][2]);//z +- 1or0

                            if (classCurrent != classSurrounding)
                            {
                                Points.Add(currentSample);
                                break;
                            }
                        }
                    }

            _scatterPlot1.AddPoints(Points, 2, false);
            Points.Clear();
            //-----------------------------------------------------------------------------------------------------------
            //draw training data
            for (int j = 0; j < 3; j++)
            {
                for (int i = 0; i < DATA_dum.Length; i++)
                {
                    if (DATA_dum[i][j + 3] == 1)
                    {
                        Points.Add(new double[] { DATA_dum[i][0], DATA_dum[i][1], DATA_dum[i][2] });
                    }
                }
                _scatterPlot1.AddPoints(Points, 6);
                Points.Clear();
            }

            //-----------------------------------------------------------------------------------------------------------
            //draw fail training data
            for (int i = 0; i < failCount; i++)
                Points.Add(new double[] { DATA_fail[i][0], DATA_fail[i][1], DATA_fail[i][2] });

            _scatterPlot1.AddPoints(Points, 6, false);
            Points.Clear();
            //-----------------------------------------------------------------------------------------------------------
            //draw x and y and z

            //x base line
            Points.Add(new double[] { 0, 0, 0 });
            Points.Add(new double[] { 5, 0, 0 });
            _scatterPlot1.AddPoints(Points, -111, false);
            Points.Clear();

            //y base line
            Points.Add(new double[] { 0, 0, 0 });
            Points.Add(new double[] { 0, 2, 0 });
            _scatterPlot1.AddPoints(Points, -111, false);
            Points.Clear();

            //z base line
            Points.Add(new double[] { 0, 0, 0 });
            Points.Add(new double[] { 0, 0, 1 });
            _scatterPlot1.AddPoints(Points, -111, false);
            Points.Clear();

            //-----------------------------------------------------------------------------------------------------------
            //for (int i = 0; i < 400; i++)
            //{
            //    double R = 2;
            //    if (i > 199)
            //    {
            //        if (i == 200)
            //        {
            //            _scatterPlot1.AddPoints(Points, 3);
            //            Points.Clear();
            //        }
            //        R /= 4;
            //    }
            //    double theta = Math.PI * (rand.NextDouble()/2);
            //    double phi = 2 * Math.PI * rand.NextDouble();
            //    double x = R * Math.Sin(theta) * Math.Cos(phi);
            //    double y = R * Math.Sin(theta) * Math.Sin(phi);
            //    double z = R * Math.Cos(theta);

            //    Points.Add(new double[] { x, y, z });
            //}
            //_scatterPlot1.AddPoints(Points,3);
            //Points.Clear();
            ////-----------------------------------------------------------------------------------------------------------
            //for (int i = 0; i < 200; i++)
            //{
            //    double R = 1;
            //    if (i > 100)
            //        R /= 2;
            //    double theta = 10D / 180 * Math.PI * Math.Sin(10 * 2 * Math.PI * i / 200);
            //    double phi = 2 * Math.PI * i / 200;
            //    double x = R * Math.Cos(theta) * Math.Cos(phi);
            //    double y = R * Math.Cos(theta) * Math.Sin(phi);
            //    double z = R * Math.Sin(theta);
            //    Points.Add(new double[] { x, y, z });

            //}
            //scatterPlot1.AddPoints(Points);
            //Points.Clear();
            ////-----------------------------------------------------------------------------------------------------------
        }
        // ----------------------------------------------------------------------------------------
        public Form1()
        {
            AllocConsole();

            InitializeComponent();
            InitializeComponent_ScatterPlot();
            this.Resize += Form1_Resize;

            Draw();
        }
        // ----------------------------------------------------------------------------------------
        private void InitializeComponent_ScatterPlot()
        {
            // scatterPlot1
            this._scatterPlot1 = new ScatterPlot();
            this._scatterPlot1.Location = new Point(0, 0);
            this.Form1_Resize(null, null);
            this._scatterPlot1.Name = "scatterPlot1";


            this.Controls.Add(this._scatterPlot1);
        }
        // ----------------------------------------------------------------------------------------
        private void Form1_Resize(object sender, System.EventArgs e)
        {
            this._scatterPlot1.Size = new Size(this.Width, this.Height);
        }
        // ----------------------------------------------------------------------------------------
    }
}


namespace NeuralNetwork
{
    public class P//Program
    {
        static NN _nn;

        // ----------------------------------------------------------------------------------------
        static public void TrainAndTestTrainData(double[][] allData, double[][] failData, int numInput, int numHidden, int numOutput, ref int failCount)
        {
            _nn = new NN(numInput, numHidden, numOutput);
            _nn.InitializeWeights();

            int maxEpochs = 2000;
            double learnRate = 0.05;
            double momentum = 0.01;
            double weightDecay = 0.0001;
            double[][] trainData = null;

            MakeTrainTest(allData, out trainData);
            _nn.Train(trainData, maxEpochs, learnRate, momentum, weightDecay);

            failCount = 0;
            for (int i = 0; i < allData.Length; i++)
            {
                int cls = Classify(allData[i][0], allData[i][1], allData[i][2]);
                if (allData[i][3 + cls] != 1)
                {
                    failCount++;
                    failData[failCount - 1] = new double[] { allData[i][0], allData[i][1], allData[i][2] };
                    Console.Write("train data location: " + allData[i][0] + ", " + allData[i][1] + ", " + allData[i][2] + "  ");
                    if (allData[i][3] == 1)
                        Console.WriteLine("mis-classified from " + 0 + " to " + cls);
                    if (allData[i][4] == 1)
                        Console.WriteLine("mis-classified from " + 1 + " to " + cls);
                    if (allData[i][5] == 1)
                        Console.WriteLine("mis-classified from " + 2 + " to " + cls);
                }
            }
        }
        // ----------------------------------------------------------------------------------------
        public static int Classify(double x, double y, double z)
        {
            double[] testData = new double[] { x, y, z };
            return Classify(testData);
        }
        // ----------------------------------------------------------------------------------------
        public static int Classify(double[] testData)
        {
            return _nn.Classify(testData);
        }
        // ----------------------------------------------------------------------------------------
        static void MakeTrainTest(double[][] allData, out double[][] trainData)
        {
            // split allData into 80% trainData and 20% testData
            Random rnd = new Random(0);
            int totRows = allData.Length;
            int numCols = allData[0].Length;

            int trainRows = (int)(totRows * 0.80); // hard-coded 80-20 split
            int testRows = totRows - trainRows;

            trainData = new double[trainRows][];

            int[] sequence = new int[totRows]; // create a random sequence of indexes
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }

            int si = 0; // index into sequence[]
            int j = 0; // index into trainData or testData

            for (; si < trainRows; ++si) // first rows to train data
            {
                trainData[j] = new double[numCols];
                int idx = sequence[si];
                Array.Copy(allData[idx], trainData[j], numCols);
                ++j;
            }
        }
        // ----------------------------------------------------------------------------------------
    }

    public class NN
    {
        private static Random _rnd;

        private int _numInput;
        private int _numHidden;
        private int _numOutput;

        private double[] _inputs;

        private double[][] _ihWeights; // input-hidden
        private double[] _hBiases;
        private double[] _hOutputs;

        private double[][] _hoWeights; // hidden-output
        private double[] _oBiases;

        private double[] _outputs;

        // back-prop specific arrays (these could be local to method UpdateWeights)
        private double[] _oGrads; // output gradients for back-propagation
        private double[] _hGrads; // hidden gradients for back-propagation

        // back-prop momentum specific arrays (could be local to method Train)
        private double[][] _ihPrevWeightsDelta;  // for momentum with back-propagation
        private double[] _hPrevBiasesDelta;
        private double[][] _hoPrevWeightsDelta;
        private double[] _oPrevBiasesDelta;

        // ----------------------------------------------------------------------------------------
        public NN(int numInput, int numHidden, int numOutput)
        {
            _rnd = new Random(0); // for InitializeWeights() and Shuffle()

            this._numInput = numInput;
            this._numHidden = numHidden;
            this._numOutput = numOutput;

            this._inputs = new double[numInput];

            this._ihWeights = MakeMatrix(numInput, numHidden);
            this._hBiases = new double[numHidden];
            this._hOutputs = new double[numHidden];

            this._hoWeights = MakeMatrix(numHidden, numOutput);
            this._oBiases = new double[numOutput];

            this._outputs = new double[numOutput];

            // back-prop related arrays below
            this._hGrads = new double[numHidden];
            this._oGrads = new double[numOutput];

            this._ihPrevWeightsDelta = MakeMatrix(numInput, numHidden);
            this._hPrevBiasesDelta = new double[numHidden];
            this._hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput);
            this._oPrevBiasesDelta = new double[numOutput];
        } // ctor
        // ----------------------------------------------------------------------------------------
        private static double[][] MakeMatrix(int rows, int cols) // helper for ctor
        {
            double[][] result = new double[rows][];
            for (int r = 0; r < result.Length; ++r)
                result[r] = new double[cols];
            return result;
        }
        // ----------------------------------------------------------------------------------------
        public void SetWeights(double[] weights)
        {
            // copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
            int numWeights = (_numInput * _numHidden) + (_numHidden * _numOutput) + _numHidden + _numOutput;
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array length: ");

            int k = 0; // points into weights param

            for (int i = 0; i < _numInput; ++i)
                for (int j = 0; j < _numHidden; ++j)
                    _ihWeights[i][j] = weights[k++];
            for (int i = 0; i < _numHidden; ++i)
                _hBiases[i] = weights[k++];
            for (int i = 0; i < _numHidden; ++i)
                for (int j = 0; j < _numOutput; ++j)
                    _hoWeights[i][j] = weights[k++];
            for (int i = 0; i < _numOutput; ++i)
                _oBiases[i] = weights[k++];
        }
        // ----------------------------------------------------------------------------------------
        public void InitializeWeights()
        {
            // initialize weights and biases to small random values
            int numWeights = (_numInput * _numHidden) + (_numHidden * _numOutput) + _numHidden + _numOutput;
            double[] initialWeights = new double[numWeights];
            double lo = -0.01;
            double hi = 0.01;
            for (int i = 0; i < initialWeights.Length; ++i)
                initialWeights[i] = (hi - lo) * _rnd.NextDouble() + lo;
            this.SetWeights(initialWeights);
        }
        // ----------------------------------------------------------------------------------------
        private double[] ComputeOutputs(double[] xValues)
        {
            if (xValues.Length != _numInput)
                throw new Exception("Bad xValues array length");

            double[] hSums = new double[_numHidden]; // hidden nodes sums scratch array
            double[] oSums = new double[_numOutput]; // output nodes sums

            for (int i = 0; i < xValues.Length; ++i) // copy x-values to inputs
                this._inputs[i] = xValues[i];

            for (int j = 0; j < _numHidden; ++j)  // compute i-h sum of weights * inputs
                for (int i = 0; i < _numInput; ++i)
                    hSums[j] += this._inputs[i] * this._ihWeights[i][j]; // note +=

            for (int i = 0; i < _numHidden; ++i)  // add biases to input-to-hidden sums
                hSums[i] += this._hBiases[i];

            for (int i = 0; i < _numHidden; ++i)   // apply activation
                this._hOutputs[i] = HyperTanFunction(hSums[i]); // hard-coded

            for (int j = 0; j < _numOutput; ++j)   // compute h-o sum of weights * hOutputs
                for (int i = 0; i < _numHidden; ++i)
                    oSums[j] += _hOutputs[i] * _hoWeights[i][j];

            for (int i = 0; i < _numOutput; ++i)  // add biases to input-to-hidden sums
                oSums[i] += _oBiases[i];

            double[] softOut = Softmax(oSums); // softmax activation does all outputs at once for efficiency
            Array.Copy(softOut, _outputs, softOut.Length);

            double[] retResult = new double[_numOutput]; // could define a GetOutputs method instead
            Array.Copy(this._outputs, retResult, retResult.Length);
            return retResult;
        } // ComputeOutputs
        // ----------------------------------------------------------------------------------------
        private static double HyperTanFunction(double x)
        {
            if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
            else if (x > 20.0) return 1.0;
            else return Math.Tanh(x);
        }
        // ----------------------------------------------------------------------------------------
        public void Train(double[][] trainData, int maxEprochs, double learnRate, double momentum, double weightDecay)
        {
            // train a back-prop style NN classifier using learning rate and momentum
            // weight decay reduces the magnitude of a weight value over time unless that value
            // is constantly increased
            int epoch = 0;
            double[] xValues = new double[_numInput]; // inputs
            double[] tValues = new double[_numOutput]; // target values

            int[] sequence = new int[trainData.Length];
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            while (epoch < maxEprochs)
            {
                double mse = MeanSquaredError(trainData);
                if (mse < 0.020) break; // consider passing value in as parameter
                //if (mse < 0.001) break; // consider passing value in as parameter

                Shuffle(sequence); // visit each training data in random order
                for (int i = 0; i < trainData.Length; ++i)
                {
                    int idx = sequence[i];
                    Array.Copy(trainData[idx], xValues, _numInput);
                    Array.Copy(trainData[idx], _numInput, tValues, 0, _numOutput);
                    ComputeOutputs(xValues); // copy xValues in, compute outputs (store them internally)
                    UpdateWeights(tValues, learnRate, momentum, weightDecay); // find better weights
                } // each training tuple
                ++epoch;
            }
        } // Train
        // ----------------------------------------------------------------------------------------
        private void UpdateWeights(double[] tValues, double learnRate, double momentum, double weightDecay)
        {
            // update the weights and biases using back-propagation, with target values, eta (learning rate),
            // alpha (momentum).
            // assumes that SetWeights and ComputeOutputs have been called and so all the internal arrays
            // and matrices have values (other than 0.0)
            if (tValues.Length != _numOutput)
                throw new Exception("target values not same Length as output in UpdateWeights");

            // 1. compute output gradients
            for (int i = 0; i < _oGrads.Length; ++i)
            {
                // derivative of softmax = (1 - y) * y (same as log-sigmoid)
                double derivative = (1 - _outputs[i]) * _outputs[i];
                // 'mean squared error version' includes (1-y)(y) derivative
                _oGrads[i] = derivative * (tValues[i] - _outputs[i]);
            }

            // 2. compute hidden gradients
            for (int i = 0; i < _hGrads.Length; ++i)
            {
                // derivative of tanh = (1 - y) * (1 + y)
                double derivative = (1 - _hOutputs[i]) * (1 + _hOutputs[i]);
                double sum = 0.0;
                for (int j = 0; j < _numOutput; ++j) // each hidden delta is the sum of numOutput terms
                {
                    double x = _oGrads[j] * _hoWeights[i][j];
                    sum += x;
                }
                _hGrads[i] = derivative * sum;
            }

            // 3a. update hidden weights (gradients must be computed right-to-left but weights
            // can be updated in any order)
            for (int i = 0; i < _ihWeights.Length; ++i) // 0..2 (3)
            {
                for (int j = 0; j < _ihWeights[0].Length; ++j) // 0..3 (4)
                {
                    double delta = learnRate * _hGrads[j] * _inputs[i]; // compute the new delta
                    _ihWeights[i][j] += delta; // update. note we use '+' instead of '-'. this can be very tricky.
                    // now add momentum using previous delta. on first pass old value will be 0.0 but that's OK.
                    _ihWeights[i][j] += momentum * _ihPrevWeightsDelta[i][j];
                    _ihWeights[i][j] -= (weightDecay * _ihWeights[i][j]); // weight decay
                    _ihPrevWeightsDelta[i][j] = delta; // don't forget to save the delta for momentum 
                }
            }

            // 3b. update hidden biases
            for (int i = 0; i < _hBiases.Length; ++i)
            {
                double delta = learnRate * _hGrads[i] * 1.0; // t1.0 is constant input for bias; could leave out
                _hBiases[i] += delta;
                _hBiases[i] += momentum * _hPrevBiasesDelta[i]; // momentum
                _hBiases[i] -= (weightDecay * _hBiases[i]); // weight decay
                _hPrevBiasesDelta[i] = delta; // don't forget to save the delta
            }

            // 4. update hidden-output weights
            for (int i = 0; i < _hoWeights.Length; ++i)
            {
                for (int j = 0; j < _hoWeights[0].Length; ++j)
                {
                    // see above: hOutputs are inputs to the nn outputs
                    double delta = learnRate * _oGrads[j] * _hOutputs[i];
                    _hoWeights[i][j] += delta;
                    _hoWeights[i][j] += momentum * _hoPrevWeightsDelta[i][j]; // momentum
                    _hoWeights[i][j] -= (weightDecay * _hoWeights[i][j]); // weight decay
                    _hoPrevWeightsDelta[i][j] = delta; // save
                }
            }

            // 4b. update output biases
            for (int i = 0; i < _oBiases.Length; ++i)
            {
                double delta = learnRate * _oGrads[i] * 1.0;
                _oBiases[i] += delta;
                _oBiases[i] += momentum * _oPrevBiasesDelta[i]; // momentum
                _oBiases[i] -= (weightDecay * _oBiases[i]); // weight decay
                _oPrevBiasesDelta[i] = delta; // save
            }
        } // UpdateWeights
        // ----------------------------------------------------------------------------------------
        private static void Shuffle(int[] sequence)
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = _rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }
        // ----------------------------------------------------------------------------------------
        private double MeanSquaredError(double[][] trainData) // used as a training stopping condition
        {
            // average squared error per training tuple
            double sumSquaredError = 0.0;
            double[] xValues = new double[_numInput]; // first numInput values in trainData
            double[] tValues = new double[_numOutput]; // last numOutput values

            // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
            for (int i = 0; i < trainData.Length; ++i)
            {
                Array.Copy(trainData[i], xValues, _numInput);
                Array.Copy(trainData[i], _numInput, tValues, 0, _numOutput); // get target values
                double[] yValues = this.ComputeOutputs(xValues); // compute output using current weights
                for (int j = 0; j < _numOutput; ++j)
                {
                    double err = tValues[j] - yValues[j];
                    sumSquaredError += err * err;
                }
            }
            return sumSquaredError / trainData.Length;
        }
        // ----------------------------------------------------------------------------------------
        private static double[] Softmax(double[] oSums)
        {
            // determine max output sum
            // does all output nodes at once so scale doesn't have to be re-computed each time
            double max = oSums[0];
            for (int i = 0; i < oSums.Length; ++i)
                if (oSums[i] > max) max = oSums[i];

            // determine scaling factor -- sum of exp(each val - max)
            double scale = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
                scale += Math.Exp(oSums[i] - max);

            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i] - max) / scale;

            return result; // now scaled so that xi sum to 1.0
        }
        // ----------------------------------------------------------------------------------------
        public int Classify(double[] testData)
        {
            double[] xValues = new double[_numInput]; // inputs
            double[] yValues; // computed Y

            Array.Copy(testData, xValues, _numInput); // parse test data into x-values and t-values

            yValues = this.ComputeOutputs(xValues);

            int maxIndex = 0;
            for (int index = 0; index < yValues.Length - 1; index++)
                if (yValues[index] < yValues[index + 1])
                    maxIndex = index + 1;

            return maxIndex;//return what color
        }
        // ----------------------------------------------------------------------------------------
    } // NeuralNetwork
}










namespace Data
{
    public class data
    {
        public double[][] data1 = new double[150][];
        public double[][] data2 = new double[500][];

        public data()
        {
            data1[0] = new double[] 
                { 
                    5.1, // sepal length
                    3.5,     // sepal width
                    1.4,     // petal length
                    0.2,     // petal width
                    0,       // type1
                    0,       // type2
                    1        // type3
                };
            data1[1] = new double[] { 4.9, 3.0, 1.4, 0.2, 0, 0, 1 };
            data1[2] = new double[] { 4.7, 3.2, 1.3, 0.2, 0, 0, 1 };
            data1[3] = new double[] { 4.6, 3.1, 1.5, 0.2, 0, 0, 1 };
            data1[4] = new double[] { 5.0, 3.6, 1.4, 0.2, 0, 0, 1 };
            data1[5] = new double[] { 5.4, 3.9, 1.7, 0.4, 0, 0, 1 };
            data1[6] = new double[] { 4.6, 3.4, 1.4, 0.3, 0, 0, 1 };
            data1[7] = new double[] { 5.0, 3.4, 1.5, 0.2, 0, 0, 1 };
            data1[8] = new double[] { 4.4, 2.9, 1.4, 0.2, 0, 0, 1 };
            data1[9] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };

            data1[10] = new double[] { 5.4, 3.7, 1.5, 0.2, 0, 0, 1 };
            data1[11] = new double[] { 4.8, 3.4, 1.6, 0.2, 0, 0, 1 };
            data1[12] = new double[] { 4.8, 3.0, 1.4, 0.1, 0, 0, 1 };
            data1[13] = new double[] { 4.3, 3.0, 1.1, 0.1, 0, 0, 1 };
            data1[14] = new double[] { 5.8, 4.0, 1.2, 0.2, 0, 0, 1 };
            data1[15] = new double[] { 5.7, 4.4, 1.5, 0.4, 0, 0, 1 };
            data1[16] = new double[] { 5.4, 3.9, 1.3, 0.4, 0, 0, 1 };
            data1[17] = new double[] { 5.1, 3.5, 1.4, 0.3, 0, 0, 1 };
            data1[18] = new double[] { 5.7, 3.8, 1.7, 0.3, 0, 0, 1 };
            data1[19] = new double[] { 5.1, 3.8, 1.5, 0.3, 0, 0, 1 };

            data1[20] = new double[] { 5.4, 3.4, 1.7, 0.2, 0, 0, 1 };
            data1[21] = new double[] { 5.1, 3.7, 1.5, 0.4, 0, 0, 1 };
            data1[22] = new double[] { 4.6, 3.6, 1.0, 0.2, 0, 0, 1 };
            data1[23] = new double[] { 5.1, 3.3, 1.7, 0.5, 0, 0, 1 };
            data1[24] = new double[] { 4.8, 3.4, 1.9, 0.2, 0, 0, 1 };
            data1[25] = new double[] { 5.0, 3.0, 1.6, 0.2, 0, 0, 1 };
            data1[26] = new double[] { 5.0, 3.4, 1.6, 0.4, 0, 0, 1 };
            data1[27] = new double[] { 5.2, 3.5, 1.5, 0.2, 0, 0, 1 };
            data1[28] = new double[] { 5.2, 3.4, 1.4, 0.2, 0, 0, 1 };
            data1[29] = new double[] { 4.7, 3.2, 1.6, 0.2, 0, 0, 1 };

            data1[30] = new double[] { 4.8, 3.1, 1.6, 0.2, 0, 0, 1 };
            data1[31] = new double[] { 5.4, 3.4, 1.5, 0.4, 0, 0, 1 };
            data1[32] = new double[] { 5.2, 4.1, 1.5, 0.1, 0, 0, 1 };
            data1[33] = new double[] { 5.5, 4.2, 1.4, 0.2, 0, 0, 1 };
            data1[34] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
            data1[35] = new double[] { 5.0, 3.2, 1.2, 0.2, 0, 0, 1 };
            data1[36] = new double[] { 5.5, 3.5, 1.3, 0.2, 0, 0, 1 };
            data1[37] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
            data1[38] = new double[] { 4.4, 3.0, 1.3, 0.2, 0, 0, 1 };
            data1[39] = new double[] { 5.1, 3.4, 1.5, 0.2, 0, 0, 1 };

            data1[40] = new double[] { 5.0, 3.5, 1.3, 0.3, 0, 0, 1 };
            data1[41] = new double[] { 4.5, 2.3, 1.3, 0.3, 0, 0, 1 };
            data1[42] = new double[] { 4.4, 3.2, 1.3, 0.2, 0, 0, 1 };
            data1[43] = new double[] { 5.0, 3.5, 1.6, 0.6, 0, 0, 1 };
            data1[44] = new double[] { 5.1, 3.8, 1.9, 0.4, 0, 0, 1 };
            data1[45] = new double[] { 4.8, 3.0, 1.4, 0.3, 0, 0, 1 };
            data1[46] = new double[] { 5.1, 3.8, 1.6, 0.2, 0, 0, 1 };
            data1[47] = new double[] { 4.6, 3.2, 1.4, 0.2, 0, 0, 1 };
            data1[48] = new double[] { 5.3, 3.7, 1.5, 0.2, 0, 0, 1 };
            data1[49] = new double[] { 5.0, 3.3, 1.4, 0.2, 0, 0, 1 };

            data1[50] = new double[] { 7.0, 3.2, 4.7, 1.4, 0, 1, 0 };
            data1[51] = new double[] { 6.4, 3.2, 4.5, 1.5, 0, 1, 0 };
            data1[52] = new double[] { 6.9, 3.1, 4.9, 1.5, 0, 1, 0 };
            data1[53] = new double[] { 5.5, 2.3, 4.0, 1.3, 0, 1, 0 };
            data1[54] = new double[] { 6.5, 2.8, 4.6, 1.5, 0, 1, 0 };
            data1[55] = new double[] { 5.7, 2.8, 4.5, 1.3, 0, 1, 0 };
            data1[56] = new double[] { 6.3, 3.3, 4.7, 1.6, 0, 1, 0 };
            data1[57] = new double[] { 4.9, 2.4, 3.3, 1.0, 0, 1, 0 };
            data1[58] = new double[] { 6.6, 2.9, 4.6, 1.3, 0, 1, 0 };
            data1[59] = new double[] { 5.2, 2.7, 3.9, 1.4, 0, 1, 0 };

            data1[60] = new double[] { 5.0, 2.0, 3.5, 1.0, 0, 1, 0 };
            data1[61] = new double[] { 5.9, 3.0, 4.2, 1.5, 0, 1, 0 };
            data1[62] = new double[] { 6.0, 2.2, 4.0, 1.0, 0, 1, 0 };
            data1[63] = new double[] { 6.1, 2.9, 4.7, 1.4, 0, 1, 0 };
            data1[64] = new double[] { 5.6, 2.9, 3.6, 1.3, 0, 1, 0 };
            data1[65] = new double[] { 6.7, 3.1, 4.4, 1.4, 0, 1, 0 };
            data1[66] = new double[] { 5.6, 3.0, 4.5, 1.5, 0, 1, 0 };
            data1[67] = new double[] { 5.8, 2.7, 4.1, 1.0, 0, 1, 0 };
            data1[68] = new double[] { 6.2, 2.2, 4.5, 1.5, 0, 1, 0 };
            data1[69] = new double[] { 5.6, 2.5, 3.9, 1.1, 0, 1, 0 };

            data1[70] = new double[] { 5.9, 3.2, 4.8, 1.8, 0, 1, 0 };
            data1[71] = new double[] { 6.1, 2.8, 4.0, 1.3, 0, 1, 0 };
            data1[72] = new double[] { 6.3, 2.5, 4.9, 1.5, 0, 1, 0 };
            data1[73] = new double[] { 6.1, 2.8, 4.7, 1.2, 0, 1, 0 };
            data1[74] = new double[] { 6.4, 2.9, 4.3, 1.3, 0, 1, 0 };
            data1[75] = new double[] { 6.6, 3.0, 4.4, 1.4, 0, 1, 0 };
            data1[76] = new double[] { 6.8, 2.8, 4.8, 1.4, 0, 1, 0 };
            data1[77] = new double[] { 6.7, 3.0, 5.0, 1.7, 0, 1, 0 };
            data1[78] = new double[] { 6.0, 2.9, 4.5, 1.5, 0, 1, 0 };
            data1[79] = new double[] { 5.7, 2.6, 3.5, 1.0, 0, 1, 0 };

            data1[80] = new double[] { 5.5, 2.4, 3.8, 1.1, 0, 1, 0 };
            data1[81] = new double[] { 5.5, 2.4, 3.7, 1.0, 0, 1, 0 };
            data1[82] = new double[] { 5.8, 2.7, 3.9, 1.2, 0, 1, 0 };
            data1[83] = new double[] { 6.0, 2.7, 5.1, 1.6, 0, 1, 0 };
            data1[84] = new double[] { 5.4, 3.0, 4.5, 1.5, 0, 1, 0 };
            data1[85] = new double[] { 6.0, 3.4, 4.5, 1.6, 0, 1, 0 };
            data1[86] = new double[] { 6.7, 3.1, 4.7, 1.5, 0, 1, 0 };
            data1[87] = new double[] { 6.3, 2.3, 4.4, 1.3, 0, 1, 0 };
            data1[88] = new double[] { 5.6, 3.0, 4.1, 1.3, 0, 1, 0 };
            data1[89] = new double[] { 5.5, 2.5, 4.0, 1.3, 0, 1, 0 };

            data1[90] = new double[] { 5.5, 2.6, 4.4, 1.2, 0, 1, 0 };
            data1[91] = new double[] { 6.1, 3.0, 4.6, 1.4, 0, 1, 0 };
            data1[92] = new double[] { 5.8, 2.6, 4.0, 1.2, 0, 1, 0 };
            data1[93] = new double[] { 5.0, 2.3, 3.3, 1.0, 0, 1, 0 };
            data1[94] = new double[] { 5.6, 2.7, 4.2, 1.3, 0, 1, 0 };
            data1[95] = new double[] { 5.7, 3.0, 4.2, 1.2, 0, 1, 0 };
            data1[96] = new double[] { 5.7, 2.9, 4.2, 1.3, 0, 1, 0 };
            data1[97] = new double[] { 6.2, 2.9, 4.3, 1.3, 0, 1, 0 };
            data1[98] = new double[] { 5.1, 2.5, 3.0, 1.1, 0, 1, 0 };
            data1[99] = new double[] { 5.7, 2.8, 4.1, 1.3, 0, 1, 0 };

            data1[100] = new double[] { 6.3, 3.3, 6.0, 2.5, 1, 0, 0 };
            data1[101] = new double[] { 5.8, 2.7, 5.1, 1.9, 1, 0, 0 };
            data1[102] = new double[] { 7.1, 3.0, 5.9, 2.1, 1, 0, 0 };
            data1[103] = new double[] { 6.3, 2.9, 5.6, 1.8, 1, 0, 0 };
            data1[104] = new double[] { 6.5, 3.0, 5.8, 2.2, 1, 0, 0 };
            data1[105] = new double[] { 7.6, 3.0, 6.6, 2.1, 1, 0, 0 };
            data1[106] = new double[] { 4.9, 2.5, 4.5, 1.7, 1, 0, 0 };
            data1[107] = new double[] { 7.3, 2.9, 6.3, 1.8, 1, 0, 0 };
            data1[108] = new double[] { 6.7, 2.5, 5.8, 1.8, 1, 0, 0 };
            data1[109] = new double[] { 7.2, 3.6, 6.1, 2.5, 1, 0, 0 };

            data1[110] = new double[] { 6.5, 3.2, 5.1, 2.0, 1, 0, 0 };
            data1[111] = new double[] { 6.4, 2.7, 5.3, 1.9, 1, 0, 0 };
            data1[112] = new double[] { 6.8, 3.0, 5.5, 2.1, 1, 0, 0 };
            data1[113] = new double[] { 5.7, 2.5, 5.0, 2.0, 1, 0, 0 };
            data1[114] = new double[] { 5.8, 2.8, 5.1, 2.4, 1, 0, 0 };
            data1[115] = new double[] { 6.4, 3.2, 5.3, 2.3, 1, 0, 0 };
            data1[116] = new double[] { 6.5, 3.0, 5.5, 1.8, 1, 0, 0 };
            data1[117] = new double[] { 7.7, 3.8, 6.7, 2.2, 1, 0, 0 };
            data1[118] = new double[] { 7.7, 2.6, 6.9, 2.3, 1, 0, 0 };
            data1[119] = new double[] { 6.0, 2.2, 5.0, 1.5, 1, 0, 0 };

            data1[120] = new double[] { 6.9, 3.2, 5.7, 2.3, 1, 0, 0 };
            data1[121] = new double[] { 5.6, 2.8, 4.9, 2.0, 1, 0, 0 };
            data1[122] = new double[] { 7.7, 2.8, 6.7, 2.0, 1, 0, 0 };
            data1[123] = new double[] { 6.3, 2.7, 4.9, 1.8, 1, 0, 0 };
            data1[124] = new double[] { 6.7, 3.3, 5.7, 2.1, 1, 0, 0 };
            data1[125] = new double[] { 7.2, 3.2, 6.0, 1.8, 1, 0, 0 };
            data1[126] = new double[] { 6.2, 2.8, 4.8, 1.8, 1, 0, 0 };
            data1[127] = new double[] { 6.1, 3.0, 4.9, 1.8, 1, 0, 0 };
            data1[128] = new double[] { 6.4, 2.8, 5.6, 2.1, 1, 0, 0 };
            data1[129] = new double[] { 7.2, 3.0, 5.8, 1.6, 1, 0, 0 };

            data1[130] = new double[] { 7.4, 2.8, 6.1, 1.9, 1, 0, 0 };
            data1[131] = new double[] { 7.9, 3.8, 6.4, 2.0, 1, 0, 0 };
            data1[132] = new double[] { 6.4, 2.8, 5.6, 2.2, 1, 0, 0 };
            data1[133] = new double[] { 6.3, 2.8, 5.1, 1.5, 1, 0, 0 };
            data1[134] = new double[] { 6.1, 2.6, 5.6, 1.4, 1, 0, 0 };
            data1[135] = new double[] { 7.7, 3.0, 6.1, 2.3, 1, 0, 0 };
            data1[136] = new double[] { 6.3, 3.4, 5.6, 2.4, 1, 0, 0 };
            data1[137] = new double[] { 6.4, 3.1, 5.5, 1.8, 1, 0, 0 };
            data1[138] = new double[] { 6.0, 3.0, 4.8, 1.8, 1, 0, 0 };
            data1[139] = new double[] { 6.9, 3.1, 5.4, 2.1, 1, 0, 0 };

            data1[140] = new double[] { 6.7, 3.1, 5.6, 2.4, 1, 0, 0 };
            data1[141] = new double[] { 6.9, 3.1, 5.1, 2.3, 1, 0, 0 };
            data1[142] = new double[] { 5.8, 2.7, 5.1, 1.9, 1, 0, 0 };
            data1[143] = new double[] { 6.8, 3.2, 5.9, 2.3, 1, 0, 0 };
            data1[144] = new double[] { 6.7, 3.3, 5.7, 2.5, 1, 0, 0 };
            data1[145] = new double[] { 6.7, 3.0, 5.2, 2.3, 1, 0, 0 };
            data1[146] = new double[] { 6.3, 2.5, 5.0, 1.9, 1, 0, 0 };
            data1[147] = new double[] { 6.5, 3.0, 5.2, 2.0, 1, 0, 0 };
            data1[148] = new double[] { 6.2, 3.4, 5.4, 2.3, 1, 0, 0 };
            data1[149] = new double[] { 5.9, 3.0, 5.1, 1.8, 1, 0, 0 };


















            data2[0] = new double[] { 3.5, 1.4, 0.2, 0, 0, 1 };
            data2[1] = new double[] { 3.0, 1.4, 0.2, 0, 0, 1 };
            data2[2] = new double[] { 3.2, 1.3, 0.2, 0, 0, 1 };
            data2[3] = new double[] { 3.1, 1.5, 0.2, 0, 0, 1 };
            data2[4] = new double[] { 3.6, 1.4, 0.2, 0, 0, 1 };
            data2[5] = new double[] { 3.9, 1.7, 0.4, 0, 0, 1 };
            data2[6] = new double[] { 3.4, 1.4, 0.3, 0, 0, 1 };
            data2[7] = new double[] { 3.4, 1.5, 0.2, 0, 0, 1 };
            data2[8] = new double[] { 2.9, 1.4, 0.2, 0, 0, 1 };
            data2[9] = new double[] { 3.1, 1.5, 0.1, 0, 0, 1 };
            data2[10] = new double[] { 3.7, 1.5, 0.2, 0, 0, 1 };
            data2[11] = new double[] { 3.4, 1.6, 0.2, 0, 0, 1 };
            data2[12] = new double[] { 3.0, 1.4, 0.1, 0, 0, 1 };
            data2[13] = new double[] { 3.0, 1.1, 0.1, 0, 0, 1 };
            data2[14] = new double[] { 4.0, 1.2, 0.2, 0, 0, 1 };
            data2[15] = new double[] { 4.4, 1.5, 0.4, 0, 0, 1 };
            data2[16] = new double[] { 3.9, 1.3, 0.4, 0, 0, 1 };
            data2[17] = new double[] { 3.5, 1.4, 0.3, 0, 0, 1 };
            data2[18] = new double[] { 3.8, 1.7, 0.3, 0, 0, 1 };
            data2[19] = new double[] { 3.8, 1.5, 0.3, 0, 0, 1 };
            data2[20] = new double[] { 3.4, 1.7, 0.2, 0, 0, 1 };
            data2[21] = new double[] { 3.7, 1.5, 0.4, 0, 0, 1 };
            data2[22] = new double[] { 3.6, 1.0, 0.2, 0, 0, 1 };
            data2[23] = new double[] { 3.3, 1.7, 0.5, 0, 0, 1 };
            data2[24] = new double[] { 3.4, 1.9, 0.2, 0, 0, 1 };
            data2[25] = new double[] { 3.0, 1.6, 0.2, 0, 0, 1 };
            data2[26] = new double[] { 3.4, 1.6, 0.4, 0, 0, 1 };
            data2[27] = new double[] { 3.5, 1.5, 0.2, 0, 0, 1 };
            data2[28] = new double[] { 3.4, 1.4, 0.2, 0, 0, 1 };
            data2[29] = new double[] { 3.2, 1.6, 0.2, 0, 0, 1 };
            data2[30] = new double[] { 3.1, 1.6, 0.2, 0, 0, 1 };
            data2[31] = new double[] { 3.4, 1.5, 0.4, 0, 0, 1 };
            data2[32] = new double[] { 4.1, 1.5, 0.1, 0, 0, 1 };
            data2[33] = new double[] { 4.2, 1.4, 0.2, 0, 0, 1 };
            data2[34] = new double[] { 3.1, 1.5, 0.1, 0, 0, 1 };
            data2[35] = new double[] { 3.2, 1.2, 0.2, 0, 0, 1 };
            data2[36] = new double[] { 3.5, 1.3, 0.2, 0, 0, 1 };
            data2[37] = new double[] { 3.1, 1.5, 0.1, 0, 0, 1 };
            data2[38] = new double[] { 3.0, 1.3, 0.2, 0, 0, 1 };
            data2[39] = new double[] { 3.4, 1.5, 0.2, 0, 0, 1 };
            data2[40] = new double[] { 3.5, 1.3, 0.3, 0, 0, 1 };
            data2[41] = new double[] { 2.3, 1.3, 0.3, 0, 0, 1 };
            data2[42] = new double[] { 3.2, 1.3, 0.2, 0, 0, 1 };
            data2[43] = new double[] { 3.5, 1.6, 0.6, 0, 0, 1 };
            data2[44] = new double[] { 3.8, 1.9, 0.4, 0, 0, 1 };
            data2[45] = new double[] { 3.0, 1.4, 0.3, 0, 0, 1 };
            data2[46] = new double[] { 3.8, 1.6, 0.2, 0, 0, 1 };
            data2[47] = new double[] { 3.2, 1.4, 0.2, 0, 0, 1 };
            data2[48] = new double[] { 3.7, 1.5, 0.2, 0, 0, 1 };
            data2[49] = new double[] { 3.3, 1.4, 0.2, 0, 0, 1 };
            data2[50] = new double[] { 3.2, 4.7, 1.4, 0, 0, 1 };
            data2[51] = new double[] { 3.2, 4.5, 1.5, 0, 0, 1 };
            data2[52] = new double[] { 3.1, 4.9, 1.5, 0, 0, 1 };
            data2[53] = new double[] { 2.3, 4.0, 1.3, 0, 0, 1 };
            data2[54] = new double[] { 2.8, 4.6, 1.5, 0, 0, 1 };
            data2[55] = new double[] { 2.8, 4.5, 1.3, 0, 0, 1 };
            data2[56] = new double[] { 3.3, 4.7, 1.6, 0, 0, 1 };
            data2[57] = new double[] { 2.4, 3.3, 1.0, 0, 0, 1 };
            data2[58] = new double[] { 2.9, 4.6, 1.3, 0, 0, 1 };
            data2[59] = new double[] { 2.7, 3.9, 1.4, 0, 0, 1 };
            data2[60] = new double[] { 2.0, 3.5, 1.0, 0, 0, 1 };
            data2[61] = new double[] { 3.0, 4.2, 1.5, 0, 0, 1 };
            data2[62] = new double[] { 2.2, 4.0, 1.0, 0, 0, 1 };
            data2[63] = new double[] { 2.9, 4.7, 1.4, 0, 0, 1 };
            data2[64] = new double[] { 2.9, 3.6, 1.3, 0, 0, 1 };
            data2[65] = new double[] { 3.1, 4.4, 1.4, 0, 0, 1 };
            data2[66] = new double[] { 3.0, 4.5, 1.5, 0, 0, 1 };
            data2[67] = new double[] { 2.7, 4.1, 1.0, 0, 0, 1 };
            data2[68] = new double[] { 2.2, 4.5, 1.5, 0, 0, 1 };
            data2[69] = new double[] { 2.5, 3.9, 1.1, 0, 0, 1 };
            data2[70] = new double[] { 3.2, 4.8, 1.8, 0, 0, 1 };
            data2[71] = new double[] { 2.8, 4.0, 1.3, 0, 0, 1 };
            data2[72] = new double[] { 2.5, 4.9, 1.5, 0, 0, 1 };
            data2[73] = new double[] { 2.8, 4.7, 1.2, 0, 0, 1 };
            data2[74] = new double[] { 2.9, 4.3, 1.3, 0, 0, 1 };
            data2[75] = new double[] { 3.0, 4.4, 1.4, 0, 0, 1 };
            data2[76] = new double[] { 2.8, 4.8, 1.4, 0, 0, 1 };
            data2[77] = new double[] { 3.0, 5.0, 1.7, 0, 0, 1 };
            data2[78] = new double[] { 2.9, 4.5, 1.5, 0, 0, 1 };
            data2[79] = new double[] { 2.6, 3.5, 1.0, 0, 0, 1 };
            data2[80] = new double[] { 2.4, 3.8, 1.1, 0, 0, 1 };
            data2[81] = new double[] { 2.4, 3.7, 1.0, 0, 0, 1 };
            data2[82] = new double[] { 2.7, 3.9, 1.2, 0, 0, 1 };
            data2[83] = new double[] { 2.7, 5.1, 1.6, 0, 0, 1 };
            data2[84] = new double[] { 3.0, 4.5, 1.5, 0, 0, 1 };
            data2[85] = new double[] { 3.4, 4.5, 1.6, 0, 0, 1 };
            data2[86] = new double[] { 3.1, 4.7, 1.5, 0, 0, 1 };
            data2[87] = new double[] { 2.3, 4.4, 1.3, 0, 0, 1 };
            data2[88] = new double[] { 3.0, 4.1, 1.3, 0, 0, 1 };
            data2[89] = new double[] { 2.5, 4.0, 1.3, 0, 0, 1 };
            data2[90] = new double[] { 2.6, 4.4, 1.2, 0, 0, 1 };
            data2[91] = new double[] { 3.0, 4.6, 1.4, 0, 0, 1 };
            data2[92] = new double[] { 2.6, 4.0, 1.2, 0, 0, 1 };
            data2[93] = new double[] { 2.3, 3.3, 1.0, 0, 0, 1 };
            data2[94] = new double[] { 2.7, 4.2, 1.3, 0, 0, 1 };
            data2[95] = new double[] { 3.0, 4.2, 1.2, 0, 0, 1 };
            data2[96] = new double[] { 2.9, 4.2, 1.3, 0, 0, 1 };
            data2[97] = new double[] { 2.9, 4.3, 1.3, 0, 0, 1 };
            data2[98] = new double[] { 2.5, 3.0, 1.1, 0, 0, 1 };
            data2[99] = new double[] { 2.8, 4.1, 1.3, 0, 0, 1 };
            data2[100] = new double[] { -0.3, 1.2, 1.6, 0, 1, 0 };
            data2[101] = new double[] { -0.2, 0.1, 2.0, 0, 1, 0 };
            data2[102] = new double[] { -0.1, 0.9, 1.8, 0, 1, 0 };
            data2[103] = new double[] { 0.3, 0.2, 2.0, 0, 1, 0 };
            data2[104] = new double[] { 0.6, 0.2, 1.9, 0, 1, 0 };
            data2[105] = new double[] { 0.8, -0.9, 1.6, 0, 1, 0 };
            data2[106] = new double[] { 1.6, 0.8, 0.9, 0, 1, 0 };
            data2[107] = new double[] { 1.9, 0.6, 0.2, 0, 1, 0 };
            data2[108] = new double[] { -0.1, 0.2, 2.0, 0, 1, 0 };
            data2[109] = new double[] { 1.5, -0.1, 1.3, 0, 1, 0 };
            data2[110] = new double[] { 1.3, -1.5, 0.3, 0, 1, 0 };
            data2[111] = new double[] { -0.4, -1.4, 1.4, 0, 1, 0 };
            data2[112] = new double[] { 0.0, 0.0, 2.0, 0, 1, 0 };
            data2[113] = new double[] { -0.1, 0.1, 2.0, 0, 1, 0 };
            data2[114] = new double[] { 0.0, 0.0, 2.0, 0, 1, 0 };
            data2[115] = new double[] { 0.4, 1.4, 1.4, 0, 1, 0 };
            data2[116] = new double[] { -0.5, 1.9, 0.3, 0, 1, 0 };
            data2[117] = new double[] { 0.0, 0.0, 2.0, 0, 1, 0 };
            data2[118] = new double[] { -0.3, -0.2, 2.0, 0, 1, 0 };
            data2[119] = new double[] { 0.4, 1.9, 0.3, 0, 1, 0 };
            data2[120] = new double[] { 0.6, 0.2, 1.9, 0, 1, 0 };
            data2[121] = new double[] { -1.0, 1.4, 1.1, 0, 1, 0 };
            data2[122] = new double[] { -1.2, 0.6, 1.5, 0, 1, 0 };
            data2[123] = new double[] { -0.2, -1.6, 1.1, 0, 1, 0 };
            data2[124] = new double[] { 1.1, -1.4, 0.9, 0, 1, 0 };
            data2[125] = new double[] { -0.6, -1.8, 0.6, 0, 1, 0 };
            data2[126] = new double[] { 0.1, 0.1, 2.0, 0, 1, 0 };
            data2[127] = new double[] { -1.5, -0.7, 1.1, 0, 1, 0 };
            data2[128] = new double[] { -1.3, 1.3, 0.7, 0, 1, 0 };
            data2[129] = new double[] { 0.0, 1.3, 1.6, 0, 1, 0 };
            data2[130] = new double[] { 0.0, -0.7, 1.9, 0, 1, 0 };
            data2[131] = new double[] { 0.2, -0.2, 2.0, 0, 1, 0 };
            data2[132] = new double[] { 0.5, 1.4, 1.4, 0, 1, 0 };
            data2[133] = new double[] { -1.8, 0.9, 0.2, 0, 1, 0 };
            data2[134] = new double[] { -0.1, 0.2, 2.0, 0, 1, 0 };
            data2[135] = new double[] { -2.0, 0.1, 0.2, 0, 1, 0 };
            data2[136] = new double[] { -0.5, 1.1, 1.6, 0, 1, 0 };
            data2[137] = new double[] { 1.2, -1.4, 0.7, 0, 1, 0 };
            data2[138] = new double[] { 0.4, 0.6, 1.9, 0, 1, 0 };
            data2[139] = new double[] { -0.7, -0.4, 1.8, 0, 1, 0 };
            data2[140] = new double[] { -0.4, 1.1, 1.6, 0, 1, 0 };
            data2[141] = new double[] { 1.0, 1.4, 1.0, 0, 1, 0 };
            data2[142] = new double[] { 0.7, -1.4, 1.3, 0, 1, 0 };
            data2[143] = new double[] { 0.2, -0.5, 1.9, 0, 1, 0 };
            data2[144] = new double[] { -1.6, 0.6, 1.0, 0, 1, 0 };
            data2[145] = new double[] { -0.1, -1.9, 0.6, 0, 1, 0 };
            data2[146] = new double[] { -0.3, -0.4, 1.9, 0, 1, 0 };
            data2[147] = new double[] { -1.9, 0.6, 0.1, 0, 1, 0 };
            data2[148] = new double[] { 0.4, -1.9, 0.2, 0, 1, 0 };
            data2[149] = new double[] { -0.1, 1.9, 0.5, 0, 1, 0 };
            data2[150] = new double[] { 0.4, 1.3, 1.5, 0, 1, 0 };
            data2[151] = new double[] { -1.0, 1.4, 1.0, 0, 1, 0 };
            data2[152] = new double[] { -0.9, 1.8, 0.3, 0, 1, 0 };
            data2[153] = new double[] { -0.2, -0.8, 1.8, 0, 1, 0 };
            data2[154] = new double[] { 0.0, -1.0, 1.7, 0, 1, 0 };
            data2[155] = new double[] { -0.7, -0.8, 1.7, 0, 1, 0 };
            data2[156] = new double[] { 0.2, -0.3, 2.0, 0, 1, 0 };
            data2[157] = new double[] { 1.7, -0.8, 0.5, 0, 1, 0 };
            data2[158] = new double[] { 0.3, -0.1, 2.0, 0, 1, 0 };
            data2[159] = new double[] { -0.3, -0.9, 1.8, 0, 1, 0 };
            data2[160] = new double[] { 0.0, 0.0, 2.0, 0, 1, 0 };
            data2[161] = new double[] { -1.4, -1.2, 0.8, 0, 1, 0 };
            data2[162] = new double[] { 0.5, 1.4, 1.3, 0, 1, 0 };
            data2[163] = new double[] { 0.2, -1.0, 1.7, 0, 1, 0 };
            data2[164] = new double[] { -0.4, 0.9, 1.7, 0, 1, 0 };
            data2[165] = new double[] { -1.8, 0.7, 0.3, 0, 1, 0 };
            data2[166] = new double[] { 1.0, -0.1, 1.7, 0, 1, 0 };
            data2[167] = new double[] { 0.1, 0.3, 2.0, 0, 1, 0 };
            data2[168] = new double[] { -0.8, 1.5, 1.1, 0, 1, 0 };
            data2[169] = new double[] { 0.1, 0.7, 1.9, 0, 1, 0 };
            data2[170] = new double[] { -1.5, 0.4, 1.3, 0, 1, 0 };
            data2[171] = new double[] { 1.2, -0.7, 1.5, 0, 1, 0 };
            data2[172] = new double[] { 0.2, 0.3, 2.0, 0, 1, 0 };
            data2[173] = new double[] { -0.6, 1.5, 1.1, 0, 1, 0 };
            data2[174] = new double[] { 1.8, 0.6, 0.6, 0, 1, 0 };
            data2[175] = new double[] { 1.1, 0.0, 1.7, 0, 1, 0 };
            data2[176] = new double[] { 0.9, 1.0, 1.5, 0, 1, 0 };
            data2[177] = new double[] { -0.4, 0.6, 1.9, 0, 1, 0 };
            data2[178] = new double[] { 1.2, 0.4, 1.5, 0, 1, 0 };
            data2[179] = new double[] { -0.3, -1.9, 0.6, 0, 1, 0 };
            data2[180] = new double[] { -2.0, 0.0, 0.1, 0, 1, 0 };
            data2[181] = new double[] { 1.1, 0.1, 1.6, 0, 1, 0 };
            data2[182] = new double[] { 1.6, -1.1, 0.5, 0, 1, 0 };
            data2[183] = new double[] { 0.6, 0.7, 1.8, 0, 1, 0 };
            data2[184] = new double[] { -1.4, -1.2, 0.8, 0, 1, 0 };
            data2[185] = new double[] { 0.0, -0.9, 1.8, 0, 1, 0 };
            data2[186] = new double[] { 1.6, 0.9, 0.8, 0, 1, 0 };
            data2[187] = new double[] { -0.5, -0.2, 1.9, 0, 1, 0 };
            data2[188] = new double[] { 0.9, -1.5, 1.0, 0, 1, 0 };
            data2[189] = new double[] { -1.0, 1.1, 1.4, 0, 1, 0 };
            data2[190] = new double[] { 1.5, 0.3, 1.2, 0, 1, 0 };
            data2[191] = new double[] { -0.4, 2.0, 0.1, 0, 1, 0 };
            data2[192] = new double[] { 1.7, -1.0, 0.1, 0, 1, 0 };
            data2[193] = new double[] { 0.7, 1.9, 0.1, 0, 1, 0 };
            data2[194] = new double[] { 0.3, -1.8, 0.8, 0, 1, 0 };
            data2[195] = new double[] { -1.6, -0.9, 0.7, 0, 1, 0 };
            data2[196] = new double[] { -1.5, 0.0, 1.4, 0, 1, 0 };
            data2[197] = new double[] { 0.7, 0.2, 1.9, 0, 1, 0 };
            data2[198] = new double[] { -0.2, 0.1, 2.0, 0, 1, 0 };
            data2[199] = new double[] { 0.1, 0.2, 2.0, 0, 1, 0 };
            data2[200] = new double[] { -1.9, 0.2, 0.4, 0, 1, 0 };
            data2[201] = new double[] { 0.5, 1.5, 1.2, 0, 1, 0 };
            data2[202] = new double[] { -1.4, -1.2, 0.7, 0, 1, 0 };
            data2[203] = new double[] { -0.7, 0.5, 1.8, 0, 1, 0 };
            data2[204] = new double[] { -0.9, -1.8, 0.1, 0, 1, 0 };
            data2[205] = new double[] { -1.4, -1.1, 1.0, 0, 1, 0 };
            data2[206] = new double[] { -1.4, 0.6, 1.3, 0, 1, 0 };
            data2[207] = new double[] { 0.5, -0.1, 1.9, 0, 1, 0 };
            data2[208] = new double[] { 1.1, 0.9, 1.3, 0, 1, 0 };
            data2[209] = new double[] { 0.2, -0.1, 2.0, 0, 1, 0 };
            data2[210] = new double[] { 1.4, 1.4, 0.2, 0, 1, 0 };
            data2[211] = new double[] { 1.1, -0.3, 1.7, 0, 1, 0 };
            data2[212] = new double[] { -0.5, -1.9, 0.5, 0, 1, 0 };
            data2[213] = new double[] { 0.1, 0.7, 1.9, 0, 1, 0 };
            data2[214] = new double[] { 1.4, 1.3, 0.5, 0, 1, 0 };
            data2[215] = new double[] { 0.3, 1.4, 1.4, 0, 1, 0 };
            data2[216] = new double[] { -0.7, 1.7, 0.7, 0, 1, 0 };
            data2[217] = new double[] { 1.2, 0.7, 1.4, 0, 1, 0 };
            data2[218] = new double[] { 0.2, -1.3, 1.5, 0, 1, 0 };
            data2[219] = new double[] { -0.8, 1.2, 1.4, 0, 1, 0 };
            data2[220] = new double[] { -1.6, -0.7, 1.1, 0, 1, 0 };
            data2[221] = new double[] { 0.6, 0.3, 1.9, 0, 1, 0 };
            data2[222] = new double[] { 0.2, 0.3, 2.0, 0, 1, 0 };
            data2[223] = new double[] { 0.1, 0.1, 2.0, 0, 1, 0 };
            data2[224] = new double[] { -0.3, 1.0, 1.7, 0, 1, 0 };
            data2[225] = new double[] { 1.4, -0.3, 1.4, 0, 1, 0 };
            data2[226] = new double[] { 0.7, 0.9, 1.6, 0, 1, 0 };
            data2[227] = new double[] { -0.1, -0.8, 1.8, 0, 1, 0 };
            data2[228] = new double[] { -0.2, -0.2, 2.0, 0, 1, 0 };
            data2[229] = new double[] { 1.0, 0.3, 1.7, 0, 1, 0 };
            data2[230] = new double[] { 1.1, 0.4, 1.7, 0, 1, 0 };
            data2[231] = new double[] { 1.0, -1.2, 1.3, 0, 1, 0 };
            data2[232] = new double[] { -0.6, 0.2, 1.9, 0, 1, 0 };
            data2[233] = new double[] { 0.2, 1.9, 0.5, 0, 1, 0 };
            data2[234] = new double[] { 1.8, 0.8, 0.2, 0, 1, 0 };
            data2[235] = new double[] { 0.1, -1.4, 1.4, 0, 1, 0 };
            data2[236] = new double[] { -0.3, -1.6, 1.2, 0, 1, 0 };
            data2[237] = new double[] { -0.6, 0.6, 1.8, 0, 1, 0 };
            data2[238] = new double[] { -0.7, 1.8, 0.3, 0, 1, 0 };
            data2[239] = new double[] { -0.9, -1.7, 0.4, 0, 1, 0 };
            data2[240] = new double[] { 1.1, 1.4, 1.0, 0, 1, 0 };
            data2[241] = new double[] { -1.9, -0.5, 0.4, 0, 1, 0 };
            data2[242] = new double[] { -1.6, -0.3, 1.2, 0, 1, 0 };
            data2[243] = new double[] { -1.5, -1.3, 0.5, 0, 1, 0 };
            data2[244] = new double[] { 0.0, -0.1, 2.0, 0, 1, 0 };
            data2[245] = new double[] { 1.1, -1.7, 0.1, 0, 1, 0 };
            data2[246] = new double[] { -1.2, 0.3, 1.6, 0, 1, 0 };
            data2[247] = new double[] { -1.7, -1.1, 0.1, 0, 1, 0 };
            data2[248] = new double[] { 0.0, 0.0, 2.0, 0, 1, 0 };
            data2[249] = new double[] { -1.2, -1.6, 0.4, 0, 1, 0 };
            data2[250] = new double[] { 0.3, -0.9, 1.8, 0, 1, 0 };
            data2[251] = new double[] { -0.1, 0.5, 1.9, 0, 1, 0 };
            data2[252] = new double[] { 0.1, -1.9, 0.6, 0, 1, 0 };
            data2[253] = new double[] { -0.5, -1.0, 1.7, 0, 1, 0 };
            data2[254] = new double[] { 1.5, -1.2, 0.4, 0, 1, 0 };
            data2[255] = new double[] { 0.6, -1.9, 0.5, 0, 1, 0 };
            data2[256] = new double[] { -0.7, -1.7, 0.8, 0, 1, 0 };
            data2[257] = new double[] { 0.1, 1.1, 1.7, 0, 1, 0 };
            data2[258] = new double[] { 0.5, 0.1, 1.9, 0, 1, 0 };
            data2[259] = new double[] { -0.4, -0.7, 1.8, 0, 1, 0 };
            data2[260] = new double[] { 0.2, 1.6, 1.1, 0, 1, 0 };
            data2[261] = new double[] { 0.3, 0.5, 1.9, 0, 1, 0 };
            data2[262] = new double[] { -0.2, -0.2, 2.0, 0, 1, 0 };
            data2[263] = new double[] { 0.7, 0.7, 1.7, 0, 1, 0 };
            data2[264] = new double[] { -0.6, -1.7, 0.8, 0, 1, 0 };
            data2[265] = new double[] { -1.1, -0.3, 1.7, 0, 1, 0 };
            data2[266] = new double[] { 0.3, 1.2, 1.6, 0, 1, 0 };
            data2[267] = new double[] { 0.7, -0.8, 1.7, 0, 1, 0 };
            data2[268] = new double[] { -1.7, -1.1, 0.0, 0, 1, 0 };
            data2[269] = new double[] { 1.0, -1.5, 0.8, 0, 1, 0 };
            data2[270] = new double[] { 0.2, 0.4, 1.9, 0, 1, 0 };
            data2[271] = new double[] { 1.1, 1.5, 0.8, 0, 1, 0 };
            data2[272] = new double[] { 0.8, 1.8, 0.2, 0, 1, 0 };
            data2[273] = new double[] { 0.0, -0.3, 2.0, 0, 1, 0 };
            data2[274] = new double[] { 0.0, -0.3, 2.0, 0, 1, 0 };
            data2[275] = new double[] { 1.2, -0.1, 1.6, 0, 1, 0 };
            data2[276] = new double[] { 0.1, 0.3, 2.0, 0, 1, 0 };
            data2[277] = new double[] { 0.1, 0.1, 2.0, 0, 1, 0 };
            data2[278] = new double[] { 1.6, 0.0, 1.1, 0, 1, 0 };
            data2[279] = new double[] { 0.4, 0.5, 1.9, 0, 1, 0 };
            data2[280] = new double[] { 0.1, 1.2, 1.6, 0, 1, 0 };
            data2[281] = new double[] { 1.0, 1.7, 0.3, 0, 1, 0 };
            data2[282] = new double[] { -0.4, 1.4, 1.4, 0, 1, 0 };
            data2[283] = new double[] { -1.0, -0.3, 1.7, 0, 1, 0 };
            data2[284] = new double[] { -1.0, -0.1, 1.7, 0, 1, 0 };
            data2[285] = new double[] { -0.7, -0.4, 1.9, 0, 1, 0 };
            data2[286] = new double[] { -0.7, 0.5, 1.8, 0, 1, 0 };
            data2[287] = new double[] { -1.4, 0.4, 1.4, 0, 1, 0 };
            data2[288] = new double[] { 1.3, 0.3, 1.5, 0, 1, 0 };
            data2[289] = new double[] { 0.6, -0.4, 1.9, 0, 1, 0 };
            data2[290] = new double[] { -1.1, -1.1, 1.3, 0, 1, 0 };
            data2[291] = new double[] { 0.1, 0.0, 2.0, 0, 1, 0 };
            data2[292] = new double[] { -0.7, 0.5, 1.8, 0, 1, 0 };
            data2[293] = new double[] { 0.3, 1.2, 1.5, 0, 1, 0 };
            data2[294] = new double[] { 0.0, 0.5, 1.9, 0, 1, 0 };
            data2[295] = new double[] { 0.8, -0.1, 1.8, 0, 1, 0 };
            data2[296] = new double[] { -1.0, -0.7, 1.6, 0, 1, 0 };
            data2[297] = new double[] { 1.1, -0.6, 1.6, 0, 1, 0 };
            data2[298] = new double[] { -0.7, -0.4, 1.9, 0, 1, 0 };
            data2[299] = new double[] { -0.7, -0.4, 1.9, 0, 1, 0 };
            data2[300] = new double[] { -0.3, -0.1, 0.4, 1, 0, 0 };
            data2[301] = new double[] { 0.2, -0.3, 0.4, 1, 0, 0 };
            data2[302] = new double[] { -0.1, -0.4, 0.2, 1, 0, 0 };
            data2[303] = new double[] { -0.4, -0.2, 0.2, 1, 0, 0 };
            data2[304] = new double[] { -0.4, -0.2, 0.3, 1, 0, 0 };
            data2[305] = new double[] { -0.1, 0.5, 0.2, 1, 0, 0 };
            data2[306] = new double[] { 0.1, -0.4, 0.3, 1, 0, 0 };
            data2[307] = new double[] { 0.1, -0.1, 0.5, 1, 0, 0 };
            data2[308] = new double[] { 0.3, 0.4, 0.1, 1, 0, 0 };
            data2[309] = new double[] { 0.3, 0.4, 0.1, 1, 0, 0 };
            data2[310] = new double[] { 0.2, 0.1, 0.4, 1, 0, 0 };
            data2[311] = new double[] { 0.0, 0.2, 0.5, 1, 0, 0 };
            data2[312] = new double[] { -0.1, -0.1, 0.5, 1, 0, 0 };
            data2[313] = new double[] { -0.4, -0.3, 0.0, 1, 0, 0 };
            data2[314] = new double[] { 0.1, -0.1, 0.5, 1, 0, 0 };
            data2[315] = new double[] { -0.3, -0.1, 0.4, 1, 0, 0 };
            data2[316] = new double[] { 0.1, 0.0, 0.5, 1, 0, 0 };
            data2[317] = new double[] { -0.2, 0.0, 0.5, 1, 0, 0 };
            data2[318] = new double[] { -0.3, 0.1, 0.3, 1, 0, 0 };
            data2[319] = new double[] { 0.0, -0.4, 0.2, 1, 0, 0 };
            data2[320] = new double[] { -0.3, -0.2, 0.4, 1, 0, 0 };
            data2[321] = new double[] { -0.2, -0.4, 0.3, 1, 0, 0 };
            data2[322] = new double[] { 0.1, 0.1, 0.5, 1, 0, 0 };
            data2[323] = new double[] { 0.0, 0.0, 0.5, 1, 0, 0 };
            data2[324] = new double[] { -0.2, 0.4, 0.2, 1, 0, 0 };
            data2[325] = new double[] { -0.4, 0.0, 0.2, 1, 0, 0 };
            data2[326] = new double[] { -0.3, -0.4, 0.2, 1, 0, 0 };
            data2[327] = new double[] { 0.0, 0.1, 0.5, 1, 0, 0 };
            data2[328] = new double[] { 0.0, 0.2, 0.4, 1, 0, 0 };
            data2[329] = new double[] { 0.4, 0.0, 0.3, 1, 0, 0 };
            data2[330] = new double[] { 0.3, 0.3, 0.2, 1, 0, 0 };
            data2[331] = new double[] { -0.1, -0.3, 0.4, 1, 0, 0 };
            data2[332] = new double[] { 0.0, 0.1, 0.5, 1, 0, 0 };
            data2[333] = new double[] { -0.1, 0.0, 0.5, 1, 0, 0 };
            data2[334] = new double[] { 0.0, -0.4, 0.3, 1, 0, 0 };
            data2[335] = new double[] { -0.3, 0.1, 0.4, 1, 0, 0 };
            data2[336] = new double[] { 0.2, 0.2, 0.4, 1, 0, 0 };
            data2[337] = new double[] { 0.1, -0.4, 0.3, 1, 0, 0 };
            data2[338] = new double[] { -0.3, -0.2, 0.4, 1, 0, 0 };
            data2[339] = new double[] { 0.1, 0.0, 0.5, 1, 0, 0 };
            data2[340] = new double[] { -0.2, 0.4, 0.3, 1, 0, 0 };
            data2[341] = new double[] { 0.3, -0.4, 0.1, 1, 0, 0 };
            data2[342] = new double[] { 0.1, -0.1, 0.5, 1, 0, 0 };
            data2[343] = new double[] { -0.4, 0.2, 0.2, 1, 0, 0 };
            data2[344] = new double[] { 0.5, -0.2, 0.1, 1, 0, 0 };
            data2[345] = new double[] { 0.0, 0.1, 0.5, 1, 0, 0 };
            data2[346] = new double[] { -0.2, -0.4, 0.1, 1, 0, 0 };
            data2[347] = new double[] { 0.0, 0.0, 0.5, 1, 0, 0 };
            data2[348] = new double[] { 0.2, -0.1, 0.5, 1, 0, 0 };
            data2[349] = new double[] { 0.3, -0.1, 0.4, 1, 0, 0 };
            data2[350] = new double[] { -0.5, 0.1, 0.1, 1, 0, 0 };
            data2[351] = new double[] { 0.1, 0.0, 0.5, 1, 0, 0 };
            data2[352] = new double[] { 0.0, 0.5, 0.0, 1, 0, 0 };
            data2[353] = new double[] { 0.0, 0.0, 0.5, 1, 0, 0 };
            data2[354] = new double[] { -0.4, 0.3, 0.0, 1, 0, 0 };
            data2[355] = new double[] { -0.1, -0.5, 0.1, 1, 0, 0 };
            data2[356] = new double[] { -0.1, -0.1, 0.5, 1, 0, 0 };
            data2[357] = new double[] { -0.2, 0.2, 0.4, 1, 0, 0 };
            data2[358] = new double[] { -0.2, 0.0, 0.5, 1, 0, 0 };
            data2[359] = new double[] { 0.0, 0.4, 0.3, 1, 0, 0 };
            data2[360] = new double[] { 0.0, -0.3, 0.4, 1, 0, 0 };
            data2[361] = new double[] { -0.3, -0.3, 0.2, 1, 0, 0 };
            data2[362] = new double[] { 0.3, 0.2, 0.3, 1, 0, 0 };
            data2[363] = new double[] { 0.0, -0.4, 0.3, 1, 0, 0 };
            data2[364] = new double[] { 0.0, 0.1, 0.5, 1, 0, 0 };
            data2[365] = new double[] { -0.4, 0.0, 0.3, 1, 0, 0 };
            data2[366] = new double[] { 0.1, 0.0, 0.5, 1, 0, 0 };
            data2[367] = new double[] { 0.0, 0.1, 0.5, 1, 0, 0 };
            data2[368] = new double[] { 0.1, -0.1, 0.5, 1, 0, 0 };
            data2[369] = new double[] { 0.5, -0.2, 0.1, 1, 0, 0 };
            data2[370] = new double[] { 0.4, 0.3, 0.1, 1, 0, 0 };
            data2[371] = new double[] { -0.3, 0.1, 0.4, 1, 0, 0 };
            data2[372] = new double[] { -0.1, 0.0, 0.5, 1, 0, 0 };
            data2[373] = new double[] { 0.0, 0.1, 0.5, 1, 0, 0 };
            data2[374] = new double[] { -0.2, -0.2, 0.4, 1, 0, 0 };
            data2[375] = new double[] { 0.0, -0.1, 0.5, 1, 0, 0 };
            data2[376] = new double[] { 0.0, 0.3, 0.4, 1, 0, 0 };
            data2[377] = new double[] { 0.2, 0.3, 0.3, 1, 0, 0 };
            data2[378] = new double[] { -0.5, 0.1, 0.2, 1, 0, 0 };
            data2[379] = new double[] { 0.2, -0.3, 0.4, 1, 0, 0 };
            data2[380] = new double[] { -0.1, 0.4, 0.3, 1, 0, 0 };
            data2[381] = new double[] { -0.1, -0.4, 0.3, 1, 0, 0 };
            data2[382] = new double[] { -0.3, 0.4, 0.1, 1, 0, 0 };
            data2[383] = new double[] { 0.4, 0.1, 0.3, 1, 0, 0 };
            data2[384] = new double[] { 0.2, 0.3, 0.4, 1, 0, 0 };
            data2[385] = new double[] { -0.1, 0.1, 0.5, 1, 0, 0 };
            data2[386] = new double[] { -0.4, -0.2, 0.3, 1, 0, 0 };
            data2[387] = new double[] { -0.2, 0.3, 0.4, 1, 0, 0 };
            data2[388] = new double[] { 0.0, 0.2, 0.4, 1, 0, 0 };
            data2[389] = new double[] { 0.2, 0.3, 0.3, 1, 0, 0 };
            data2[390] = new double[] { -0.4, 0.3, 0.0, 1, 0, 0 };
            data2[391] = new double[] { -0.3, 0.1, 0.4, 1, 0, 0 };
            data2[392] = new double[] { 0.1, -0.1, 0.5, 1, 0, 0 };
            data2[393] = new double[] { 0.0, -0.1, 0.5, 1, 0, 0 };
            data2[394] = new double[] { 0.4, 0.0, 0.3, 1, 0, 0 };
            data2[395] = new double[] { 0.0, -0.3, 0.4, 1, 0, 0 };
            data2[396] = new double[] { 0.3, 0.2, 0.4, 1, 0, 0 };
            data2[397] = new double[] { -0.2, -0.1, 0.4, 1, 0, 0 };
            data2[398] = new double[] { 0.0, -0.4, 0.3, 1, 0, 0 };
            data2[399] = new double[] { 0.0, -0.4, 0.3, 1, 0, 0 };
            data2[400] = new double[] { 0.4, 0.2, 0.1, 1, 0, 0 };
            data2[401] = new double[] { 0.2, 0.4, 0.2, 1, 0, 0 };
            data2[402] = new double[] { -0.1, -0.4, 0.3, 1, 0, 0 };
            data2[403] = new double[] { 0.1, -0.1, 0.5, 1, 0, 0 };
            data2[404] = new double[] { 0.2, 0.4, 0.1, 1, 0, 0 };
            data2[405] = new double[] { -0.5, -0.1, 0.1, 1, 0, 0 };
            data2[406] = new double[] { -0.1, 0.5, 0.0, 1, 0, 0 };
            data2[407] = new double[] { 0.0, 0.5, 0.1, 1, 0, 0 };
            data2[408] = new double[] { 0.4, 0.3, 0.1, 1, 0, 0 };
            data2[409] = new double[] { 0.4, 0.0, 0.3, 1, 0, 0 };
            data2[410] = new double[] { 0.0, -0.2, 0.5, 1, 0, 0 };
            data2[411] = new double[] { -0.3, 0.1, 0.4, 1, 0, 0 };
            data2[412] = new double[] { -0.3, 0.4, 0.2, 1, 0, 0 };
            data2[413] = new double[] { -0.5, 0.0, 0.1, 1, 0, 0 };
            data2[414] = new double[] { 0.1, -0.2, 0.5, 1, 0, 0 };
            data2[415] = new double[] { -0.4, -0.3, 0.1, 1, 0, 0 };
            data2[416] = new double[] { 0.1, -0.4, 0.3, 1, 0, 0 };
            data2[417] = new double[] { 0.2, -0.2, 0.4, 1, 0, 0 };
            data2[418] = new double[] { -0.1, 0.4, 0.3, 1, 0, 0 };
            data2[419] = new double[] { -0.5, 0.1, 0.0, 1, 0, 0 };
            data2[420] = new double[] { 0.1, 0.5, 0.2, 1, 0, 0 };
            data2[421] = new double[] { 0.0, 0.0, 0.5, 1, 0, 0 };
            data2[422] = new double[] { 0.1, -0.1, 0.5, 1, 0, 0 };
            data2[423] = new double[] { 0.2, 0.4, 0.1, 1, 0, 0 };
            data2[424] = new double[] { 0.2, 0.0, 0.5, 1, 0, 0 };
            data2[425] = new double[] { -0.4, 0.2, 0.2, 1, 0, 0 };
            data2[426] = new double[] { -0.1, -0.1, 0.5, 1, 0, 0 };
            data2[427] = new double[] { 0.3, 0.1, 0.4, 1, 0, 0 };
            data2[428] = new double[] { -0.2, 0.3, 0.3, 1, 0, 0 };
            data2[429] = new double[] { 0.2, 0.3, 0.4, 1, 0, 0 };
            data2[430] = new double[] { -0.3, -0.4, 0.1, 1, 0, 0 };
            data2[431] = new double[] { 0.4, -0.1, 0.3, 1, 0, 0 };
            data2[432] = new double[] { 0.0, 0.3, 0.4, 1, 0, 0 };
            data2[433] = new double[] { 0.2, 0.5, 0.1, 1, 0, 0 };
            data2[434] = new double[] { 0.5, 0.0, 0.0, 1, 0, 0 };
            data2[435] = new double[] { -0.4, -0.2, 0.3, 1, 0, 0 };
            data2[436] = new double[] { 0.1, 0.0, 0.5, 1, 0, 0 };
            data2[437] = new double[] { 0.2, 0.0, 0.5, 1, 0, 0 };
            data2[438] = new double[] { 0.4, 0.3, 0.2, 1, 0, 0 };
            data2[439] = new double[] { 0.1, -0.2, 0.4, 1, 0, 0 };
            data2[440] = new double[] { 0.0, -0.4, 0.3, 1, 0, 0 };
            data2[441] = new double[] { 0.0, 0.2, 0.5, 1, 0, 0 };
            data2[442] = new double[] { -0.1, 0.1, 0.5, 1, 0, 0 };
            data2[443] = new double[] { 0.3, 0.0, 0.4, 1, 0, 0 };
            data2[444] = new double[] { 0.0, 0.2, 0.4, 1, 0, 0 };
            data2[445] = new double[] { 0.0, 0.2, 0.5, 1, 0, 0 };
            data2[446] = new double[] { 0.1, -0.5, 0.1, 1, 0, 0 };
            data2[447] = new double[] { 0.3, 0.2, 0.3, 1, 0, 0 };
            data2[448] = new double[] { 0.4, 0.2, 0.3, 1, 0, 0 };
            data2[449] = new double[] { 0.0, 0.0, 0.5, 1, 0, 0 };
            data2[450] = new double[] { -0.4, -0.3, 0.0, 1, 0, 0 };
            data2[451] = new double[] { 0.0, 0.2, 0.5, 1, 0, 0 };
            data2[452] = new double[] { 0.2, 0.2, 0.4, 1, 0, 0 };
            data2[453] = new double[] { 0.1, 0.2, 0.5, 1, 0, 0 };
            data2[454] = new double[] { -0.2, 0.1, 0.4, 1, 0, 0 };
            data2[455] = new double[] { -0.2, 0.0, 0.4, 1, 0, 0 };
            data2[456] = new double[] { 0.0, 0.0, 0.5, 1, 0, 0 };
            data2[457] = new double[] { 0.4, 0.2, 0.3, 1, 0, 0 };
            data2[458] = new double[] { -0.2, 0.2, 0.4, 1, 0, 0 };
            data2[459] = new double[] { 0.5, -0.1, 0.1, 1, 0, 0 };
            data2[460] = new double[] { 0.4, -0.3, 0.1, 1, 0, 0 };
            data2[461] = new double[] { -0.1, -0.5, 0.1, 1, 0, 0 };
            data2[462] = new double[] { -0.4, 0.3, 0.2, 1, 0, 0 };
            data2[463] = new double[] { -0.3, 0.3, 0.2, 1, 0, 0 };
            data2[464] = new double[] { 0.2, 0.3, 0.4, 1, 0, 0 };
            data2[465] = new double[] { -0.2, -0.3, 0.4, 1, 0, 0 };
            data2[466] = new double[] { 0.0, 0.1, 0.5, 1, 0, 0 };
            data2[467] = new double[] { 0.4, -0.2, 0.1, 1, 0, 0 };
            data2[468] = new double[] { 0.4, 0.1, 0.3, 1, 0, 0 };
            data2[469] = new double[] { 0.5, 0.0, 0.1, 1, 0, 0 };
            data2[470] = new double[] { 0.0, 0.1, 0.5, 1, 0, 0 };
            data2[471] = new double[] { 0.3, -0.3, 0.3, 1, 0, 0 };
            data2[472] = new double[] { 0.1, -0.5, 0.0, 1, 0, 0 };
            data2[473] = new double[] { 0.0, 0.2, 0.5, 1, 0, 0 };
            data2[474] = new double[] { -0.1, -0.1, 0.5, 1, 0, 0 };
            data2[475] = new double[] { -0.1, 0.4, 0.2, 1, 0, 0 };
            data2[476] = new double[] { 0.2, 0.1, 0.5, 1, 0, 0 };
            data2[477] = new double[] { 0.1, 0.0, 0.5, 1, 0, 0 };
            data2[478] = new double[] { -0.5, 0.0, 0.2, 1, 0, 0 };
            data2[479] = new double[] { 0.4, 0.0, 0.3, 1, 0, 0 };
            data2[480] = new double[] { 0.2, -0.2, 0.4, 1, 0, 0 };
            data2[481] = new double[] { 0.2, 0.4, 0.2, 1, 0, 0 };
            data2[482] = new double[] { 0.0, 0.0, 0.5, 1, 0, 0 };
            data2[483] = new double[] { -0.1, -0.3, 0.4, 1, 0, 0 };
            data2[484] = new double[] { -0.1, -0.4, 0.2, 1, 0, 0 };
            data2[485] = new double[] { -0.2, 0.3, 0.3, 1, 0, 0 };
            data2[486] = new double[] { 0.2, -0.3, 0.3, 1, 0, 0 };
            data2[487] = new double[] { -0.4, -0.3, 0.2, 1, 0, 0 };
            data2[488] = new double[] { 0.2, 0.0, 0.5, 1, 0, 0 };
            data2[489] = new double[] { 0.4, -0.2, 0.2, 1, 0, 0 };
            data2[490] = new double[] { -0.1, 0.0, 0.5, 1, 0, 0 };
            data2[491] = new double[] { 0.1, -0.1, 0.5, 1, 0, 0 };
            data2[492] = new double[] { 0.2, 0.2, 0.4, 1, 0, 0 };
            data2[493] = new double[] { 0.3, 0.0, 0.4, 1, 0, 0 };
            data2[494] = new double[] { 0.0, -0.2, 0.5, 1, 0, 0 };
            data2[495] = new double[] { 0.1, 0.0, 0.5, 1, 0, 0 };
            data2[496] = new double[] { 0.2, -0.3, 0.3, 1, 0, 0 };
            data2[497] = new double[] { 0.0, 0.4, 0.3, 1, 0, 0 };
            data2[498] = new double[] { -0.5, -0.1, 0.1, 1, 0, 0 };
            data2[499] = new double[] { -0.5, -0.1, 0.1, 1, 0, 0 };

        }
    }
}

namespace Plot3D  //tool
{
    public partial class ScatterPlot : UserControl
    {
        List<List<double[]>> _Points = new List<List<double[]>>();
        List<PointF[]> _ProjPoints = new List<PointF[]>();
        int[] _ProjPointsRadius = new int[100];
        private double _f = 1000;
        private double _d = 5;
        private double[] _d_w = new double[3];
        private double _last_azimuth, _azimuth = 0, _last_elevation, _elevation = 0;
        private bool _leftMousePressed = false;
        private PointF _ptMouseClick;
        Color[] _colorIdx = new Color[] { Color.Blue, Color.Red, Color.Green, Color.Orange, Color.Fuchsia, Color.Aqua };
        static double _sumX = 0;
        static double _sumY = 0;
        static double _sumZ = 0;
        static int _count = 0;

        public double Distance
        {
            get { return _d; }
            set { _d = (value >= 0.1) ? _d = value : _d; UpdateProjection(); }
        }

        public double F
        {
            get { return _f; }
            set { _f = value; UpdateProjection(); }
        }

        public double[] CameraPos
        {
            get { return _d_w; }
            set { _d_w = value; UpdateProjection(); }
        }

        public double Azimuth
        {
            get { return _azimuth; }
            set { _azimuth = value; UpdateProjection(); }
        }

        public double Elevation
        {
            get { return _elevation; }
            set { _elevation = value; UpdateProjection(); }
        }

        public ScatterPlot()
        {
            //InitializeComponent();
            this.MouseDown += new MouseEventHandler(this.ScatterPlot_MouseDown);
            this.MouseMove += new MouseEventHandler(this.ScatterPlot_MouseMove);
            this.MouseUp += new MouseEventHandler(this.ScatterPlot_MouseUp);
            MouseWheelHandler.Add(this, MyOnMouseWheel);
        }

        protected override CreateParams CreateParams
        {
            get
            {
                var cp = base.CreateParams;
                cp.ExStyle |= 0x02000000;    // Turn on WS_EX_COMPOSITED
                return cp;
            }
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);

            Graphics g = this.CreateGraphics();
            g.FillRectangle(Brushes.Black, new Rectangle(0, 0, this.Width, this.Height));

            if (_ProjPoints != null)
                for (int i = 0; i < _ProjPoints.Count; i++)
                    foreach (PointF p in _ProjPoints[i])
                    {
                        if (_ProjPointsRadius[i] == -111)//draw coord line
                        {
                            g.DrawLine(new Pen(Color.Gray), _ProjPoints[i][0].X, _ProjPoints[i][0].Y, _ProjPoints[i][1].X, _ProjPoints[i][1].Y);
                            g.DrawLine(new Pen(Color.Gray), _ProjPoints[i][1].X, _ProjPoints[i][1].Y, _ProjPoints[i][1].X, _ProjPoints[i][1].Y);
                        }
                        else
                        {
                            g.FillEllipse(new SolidBrush(_colorIdx[i % _colorIdx.Length]), new RectangleF(p.X, p.Y, _ProjPointsRadius[i], _ProjPointsRadius[i]));
                        }
                    }

        }

        public void AddPoint(double x, double y, double z, int series)
        {
            if (_Points.Count - 1 < series)
            {
                _Points.Add(new List<double[]>());
            }

            _Points[series].Add(new double[] { x, y, z });

            foreach (List<double[]> ser in _Points)
            {
                if (_ProjPoints.Count - 1 < series)
                    _ProjPoints.Add(Projection.ProjectVector(ser, this.Width, this.Height, _f, _d_w, _azimuth, _elevation));
                else
                    _ProjPoints[series] = Projection.ProjectVector(ser, this.Width, this.Height, _f, _d_w, _azimuth, _elevation);
            }
            this.Invalidate();
        }

        public void AddPoints(List<double[]> points, int radius, bool isWeightOri = true)
        {
            if (isWeightOri)
            {
                foreach (double[] db in points)
                {
                    _sumX += db[0];
                    _sumY += db[1];
                    _sumZ += db[2];
                    _count++;
                }
            }

            List<double[]> _tmp = new List<double[]>(points);
            _Points.Add(_tmp);

            _ProjPointsRadius[_ProjPoints.Count] = radius;
            _ProjPoints.Add(Projection.ProjectVector(_Points[_Points.Count - 1], this.Width, this.Height, _f, _d_w, _azimuth, _elevation));



            UpdateProjection();
        }

        public void Clear()
        {
            _ProjPoints.Clear();
            _Points.Clear();
            Azimuth = 0;
            Elevation = 0;
        }

        private void ScatterPlot_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                _leftMousePressed = true;
                _ptMouseClick = new PointF(e.X, e.Y);
                _last_azimuth = _azimuth;
                _last_elevation = _elevation;
            }
        }

        private void ScatterPlot_MouseMove(object sender, MouseEventArgs e)
        {
            if (_leftMousePressed)
            {

                _azimuth = _last_azimuth - (_ptMouseClick.X - e.X) / 100;
                _elevation = _last_elevation + (_ptMouseClick.Y - e.Y) / 100;
                UpdateProjection();
            }
        }

        private void ScatterPlot_MouseUp(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
                _leftMousePressed = false;
        }

        private void MyOnMouseWheel(MouseEventArgs e)
        {
            Distance += -e.Delta / 500D;
        }

        private void UpdateProjection()
        {
            if (_ProjPoints == null)
                return;
            double x = _d * Math.Cos(_elevation) * Math.Cos(_azimuth);
            double y = _d * Math.Cos(_elevation) * Math.Sin(_azimuth);
            double z = _d * Math.Sin(_elevation);
            _d_w = new double[3] { 
                -y+(_sumY / _count), 
                z+(_sumZ / _count), 
                -x+(_sumX / _count) };



            for (int i = 0; i < _ProjPoints.Count; i++)
            {
                _ProjPoints[i] = Projection.ProjectVector(_Points[i], this.Width, this.Height, _f, _d_w, _azimuth, _elevation);
            }
            this.Invalidate();
        }
    }

    public static class MouseWheelHandler
    {
        public static void Add(Control ctrl, Action<MouseEventArgs> onMouseWheel)
        {
            if (ctrl == null || onMouseWheel == null)
                throw new ArgumentNullException();

            var filter = new MouseWheelMessageFilter(ctrl, onMouseWheel);
            Application.AddMessageFilter(filter);
            ctrl.Disposed += (s, e) => Application.RemoveMessageFilter(filter);
        }

        class MouseWheelMessageFilter : IMessageFilter
        {
            private readonly Control _ctrl;
            private readonly Action<MouseEventArgs> _onMouseWheel;

            public MouseWheelMessageFilter(Control ctrl, Action<MouseEventArgs> onMouseWheel)
            {
                _ctrl = ctrl;
                _onMouseWheel = onMouseWheel;
            }

            public bool PreFilterMessage(ref Message m)
            {
                var parent = _ctrl.Parent;
                if (parent != null && m.Msg == 0x20a) // WM_MOUSEWHEEL, find the control at screen position m.LParam
                {
                    var pos = new Point(m.LParam.ToInt32() & 0xffff, m.LParam.ToInt32() >> 16);

                    var clientPos = _ctrl.PointToClient(pos);

                    if (_ctrl.ClientRectangle.Contains(clientPos)
                     && ReferenceEquals(_ctrl, parent.GetChildAtPoint(parent.PointToClient(pos))))
                    {
                        var wParam = m.WParam.ToInt32();
                        Func<int, MouseButtons, MouseButtons> getButton =
                            (flag, button) => ((wParam & flag) == flag) ? button : MouseButtons.None;

                        var buttons = getButton(wParam & 0x0001, MouseButtons.Left)
                                    | getButton(wParam & 0x0010, MouseButtons.Middle)
                                    | getButton(wParam & 0x0002, MouseButtons.Right)
                                    | getButton(wParam & 0x0020, MouseButtons.XButton1)
                                    | getButton(wParam & 0x0040, MouseButtons.XButton2)
                                    ; // Not matching for these /*MK_SHIFT=0x0004;MK_CONTROL=0x0008*/

                        var delta = wParam >> 16;
                        var e = new MouseEventArgs(buttons, 0, clientPos.X, clientPos.Y, delta);
                        _onMouseWheel(e);

                        return true;
                    }
                }
                return false;
            }
        }
    }

    static class Algerbra
    {
        public class Matrix<T>
        {
            int _rows;
            int _columns;
            private T[,] _matrix;

            public Matrix(int n, int m)
            {
                _matrix = new T[n, m];
                _rows = n;
                _columns = m;
            }

            public void SetValByIdx(int m, int n, T x)
            {
                _matrix[n, m] = x;
            }

            public T GetValByIndex(int n, int m)
            {
                return _matrix[n, m];
            }

            public void SetMatrix(T[] arr)
            {
                for (int r = 0; r < _rows; r++)
                    for (int c = 0; c < _columns; c++)
                        _matrix[r, c] = arr[r * _columns + c];
            }

            public static Matrix<T> operator |(Matrix<T> m1, Matrix<T> m2)
            {
                Matrix<T> m = new Matrix<T>(m1._rows, m1._columns + m2._columns);
                for (int r = 0; r < m1._rows; r++)
                {
                    for (int c = 0; c < m1._columns; c++)
                        m._matrix[r, c] = m1._matrix[r, c];
                    for (int c = 0; c < m2._columns; c++)
                        m._matrix[r, c + m1._columns] = m2._matrix[r, c];
                }
                return m;
            }

            public static Matrix<T> operator *(Matrix<T> m1, Matrix<T> m2)
            {
                Matrix<T> m = new Matrix<T>(m1._rows, m2._columns);
                for (int r = 0; r < m._rows; r++)
                    for (int c = 0; c < m._columns; c++)
                    {
                        T tmp = (dynamic)0;
                        for (int i = 0; i < m2._rows; i++)
                            tmp += (dynamic)m1._matrix[r, i] * (dynamic)m2._matrix[i, c];
                        m._matrix[r, c] = tmp;
                    }
                return m;
            }

            public static Matrix<T> operator -(Matrix<T> m)
            {
                Matrix<T> tmp = new Matrix<T>(m._columns, m._rows);
                for (int r = 0; r < m._rows; r++)
                    for (int c = 0; c < m._columns; c++)
                        tmp._matrix[r, c] = -(dynamic)m._matrix[r, c];
                return tmp;
            }
        }
    }

    static class Projection
    {
        static public PointF Project(double[] x, double s_x, double s_y, double f, double[] d_w, double azimuth, double elevation)
        {
            Algerbra.Matrix<double> Mext = GetMext(azimuth, elevation, d_w);
            Algerbra.Matrix<double> Mint = GetMint(s_x, s_y, f);
            Algerbra.Matrix<double> X_h = new Algerbra.Matrix<double>(4, 1);
            X_h.SetMatrix(new double[] { x[0], x[1], x[2], 1.0 });
            //Debug.Print((Mint * Mext).ToString());
            Algerbra.Matrix<double> P = Mint * Mext * X_h;
            return new PointF((float)(P.GetValByIndex(0, 0) / P.GetValByIndex(2, 0)), (float)(P.GetValByIndex(1, 0) / P.GetValByIndex(2, 0)));
        }

        static public PointF[] ProjectVector(List<double[]> x, double s_x, double s_y, double f, double[] d_w, double azimuth, double elevation)
        {
            Algerbra.Matrix<double> Mext = GetMext(azimuth, elevation, d_w);
            Algerbra.Matrix<double> Mint = GetMint(s_x, s_y, f);
            Algerbra.Matrix<double> X_h = new Algerbra.Matrix<double>(4, 1);

            PointF[] Pvec = new PointF[x.Count];
            for (int i = 0; i < x.Count; i++)
            {
                X_h.SetMatrix(new double[] { x[i][0], x[i][1], x[i][2], 1.0 });
                Algerbra.Matrix<double> P = Mint * Mext * X_h;
                Pvec[i] = new PointF((float)(P.GetValByIndex(0, 0) / P.GetValByIndex(2, 0)), (float)(P.GetValByIndex(1, 0) / P.GetValByIndex(2, 0)));
            }
            return Pvec;
        }

        static Algerbra.Matrix<double> GetMint(double s_x, double s_y, double f)
        {
            Algerbra.Matrix<double> Mint = new Algerbra.Matrix<double>(3, 3);
            double o_x = s_x / 2;
            double o_y = s_y / 2;
            double a = 1;
            Mint.SetMatrix(new double[] { f, 0, o_x, 0, f * a, o_y, 0, 0, 1 });
            return Mint;
        }

        static Algerbra.Matrix<double> GetMext(double azimuth, double elevation, double[] d_w)
        {
            Algerbra.Matrix<double> R = RotationMatrix(azimuth, elevation);
            Algerbra.Matrix<double> dw = new Algerbra.Matrix<double>(3, 1);
            dw.SetMatrix(d_w);
            Algerbra.Matrix<double> Mext = R | (-R * dw);
            return Mext;
        }

        static Algerbra.Matrix<double> RotationMatrix(double azimuth, double elevation)
        {
            Algerbra.Matrix<double> R = new Algerbra.Matrix<double>(3, 3);
            R.SetMatrix(new double[] { Math.Cos(azimuth), 0, -Math.Sin(azimuth),
                                       Math.Sin(azimuth)*Math.Sin(elevation),  Math.Cos(elevation), Math.Cos(azimuth)*Math.Sin(elevation),
                                       Math.Cos(elevation)*Math.Sin(azimuth), -Math.Sin(elevation), Math.Cos(azimuth)*Math.Cos(elevation) });
            return R;
        }
    }
}
