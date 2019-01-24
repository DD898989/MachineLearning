using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Runtime.InteropServices;

namespace WindowsFormsApplication11
{
    public partial class Form1 : Form
    {
        public class SamplePoint
        {
            readonly public double dX, dY;//coord
            readonly public int nX, nY;//pixel
            readonly public int Color; //0 or 1

            public SamplePoint(double dX, double dY, int Color)
            {
                this.dX = dX;
                this.dY = dY;
                this.nX = (int)(dX * 10 + WIDTH / 2);
                this.nY = (int)(HEIGHT / 2 - dY * 10);
                this.Color = Color;
            }

            public SamplePoint(int nX, int nY, int Color)
            {
                this.nX = nX;
                this.nY = nY;
                this.dX = (double)(nX - WIDTH / 2) / 10;
                this.dY = (double)(HEIGHT / 2 - nY) / 10;
                this.Color = Color;
            }
        }

        static int WIDTH, HEIGHT;
        Graphics objGraphics;
        List<SamplePoint> ptsTrain = new List<SamplePoint>();
        List<SamplePoint> ptsTest = new List<SamplePoint>();
        static Random rnd = new Random(0);

        [DllImport("kernel32.dll", SetLastError = true)] //AllocConsole();
        [return: MarshalAs(UnmanagedType.Bool)]          //AllocConsole();
        static extern bool AllocConsole();               //AllocConsole();
        public Form1()
        {
            AllocConsole();

            InitializeComponent();
            this.panel1.BackColor = Color.White;
            this.panel1.Location = new System.Drawing.Point(0, 0);
            this.panel1.Height = this.Height;
            this.panel1.Width = this.Width;
            this.panel1.SendToBack();
            this.panel1.MouseDown += new System.Windows.Forms.MouseEventHandler(this.panel_Canvas_MouseDown);
            this.button1.Click += new System.EventHandler(this.btnRun_Click);

            objGraphics = panel1.CreateGraphics();
            WIDTH = panel1.Width;
            HEIGHT = panel1.Height;
        }
        private void Canvas_Draw_Point(int X, int Y, bool bColor, float penWid = 2, int penLen = 3, bool isSeperatePoint = false)
        {
            Pen pen;

            if (isSeperatePoint)
            {
                pen = new Pen(Color.Green);
                pen.Width = 1;
                penLen = 1;
            }
            else if (bColor)
            {
                pen = new Pen(Color.Blue);
                pen.Width = penWid;
            }
            else
            {
                pen = new Pen(Color.Red);
                pen.Width = penWid;
            }

            objGraphics.DrawLine(pen, new Point(X - penLen, Y), new Point(X + penLen, Y));
            objGraphics.DrawLine(pen, new Point(X, Y - penLen), new Point(X, Y + penLen));
        }

        private void btnRun_Click(object sender, EventArgs e)
        {
            //clear canvas
            objGraphics.Clear(Color.White);

            //redraw red and blue dots
            for (int i = 0; i < ptsTrain.Count; i++)
            {
                Canvas_Draw_Point(
                    ptsTrain[i].nX,
                    ptsTrain[i].nY,
                    Convert.ToBoolean(ptsTrain[i].Color)
                    );
            }

            //run training and see result
            TrainAndShow();
        }

        private void panel_Canvas_MouseDown(object sender, MouseEventArgs e)
        {
            SamplePoint sample;

            if (e.Button == MouseButtons.Left)
                sample = new SamplePoint(e.X, e.Y, 1);
            else
                sample = new SamplePoint(e.X, e.Y, 0);

            ptsTrain.Add(sample);

            Canvas_Draw_Point(e.X, e.Y, e.Button == MouseButtons.Left);
        }

        private void TrainAndShow()
        {
            int numTrain = ptsTrain.Count;

            //init alphas[]
            double[] weights = new double[numTrain + 1];  // one alpha weight for each train item, plus bias at end
            Array.Clear(weights, 0, weights.Length);

            //init kernelMatrix[][]       pre-compute all kernels - only viable if numTrain is not huge
            double[][] kernelMatrix = new double[numTrain][]; // item-item similarity
            for (int i = 0; i < kernelMatrix.Length; ++i)
                kernelMatrix[i] = new double[numTrain];

            //assign kernelMatrix[][]
            double sigma = 1.0;
            for (int i = 0; i < numTrain; ++i)  // pre-compute all Kernel
            {
                for (int j = 0; j < numTrain; ++j)
                {
                    double k = Kernel(ptsTrain[i], ptsTrain[j], sigma);
                    kernelMatrix[i][j] = kernelMatrix[j][i] = k;//elements in kernelMatrix are relationship between each other point,  so [j][i] = [i][j]   [i][i]=1
                }
            }

            //init and assign indices[]
            int[] indices = new int[numTrain];
            for (int i = 0; i < indices.Length; ++i)
                indices[i] = i;

            //train. aj = aj + eta(t - y) * K(i,j)
            double ETA = 0.1;  // aka learning-rate
            int MAX_ITER = 100;
            int iter = 0;

            Console.WriteLine("\nStarting training");
            Console.WriteLine("Using RBF kernel() with sigma = " + sigma.ToString("F1"));
            Console.WriteLine("Using SGD with eta = " + ETA + " and maxIter = " + MAX_ITER);

            while (iter < MAX_ITER)
            {
                Shuffle(indices);  // visit train data in random order
                for (int idx = 0; idx < indices.Length; ++idx)  // each 'from' train data
                {
                    //實際上無法在一般x,y座標上用一條線切開的兩組data
                    //在經過feature transform也就是轉成x',y'座標  就有可能切開來
                    //在perceptron是w1x1+w2x2+w0 = z   在logistic regression轉成w1x1'+w2x2'+w0 = z   
                    //一個logistic regression model = 一個Neuron     這邊一個Neuron有兩個input一個output
                    //z是x1'跟x2'的ouput   而x1'跟x2'又是另兩組Neuron(input可能是x1,x2)的output
                    //Neuron Network就是這樣串起來的
                    
                    int i = indices[idx];  // current train data index, random picked

                    double i_sum = 0.0;  // sum of alpha-i * kernel-i
                                         // sum of the relation of current training data with other training datas
                    for (int j = 0; j < weights.Length - 1; ++j)  // not the bias
                        i_sum += weights[j] * kernelMatrix[i][j];//usually, class 1 have positive weight,  class 2 have negative weight
                    i_sum += weights[weights.Length - 1];  // add bias (last alpha) -- 'input' is dummy 1.0     

                    double y = 1.0 / (1.0 + Math.Exp(-i_sum)); //sigmoid function,  if i far from any point or in the DMZ, y ~= 0.5, if i close to one of the class, y ~= 1or0;

                    //if idx==0 && iter==0     weights[...]=0,sum=0 , y=0.5

                    double i_color = ptsTrain[i].Color;  // last col holds target value

                    // update each weights
                    for (int j = 0; j < weights.Length - 1; ++j)
                        weights[j] = weights[j] + (ETA * (i_color - y) * kernelMatrix[i][j]);//i_color - y  :  0or1 - 0~1   ~~~ weight[j] + slope * learning rate * relation[i,j]

                    // update the bias
                    weights[weights.Length - 1] = weights[weights.Length - 1] + (ETA * (i_color - y)) * 1;  // dummy input
                }
                ++iter;
            }


            //display weights[]
            Console.WriteLine("\nTraining complete");
            Console.WriteLine("\nTrained model weights values: \n");
            for (int i = 0; i < weights.Length-1; ++i)
                Console.WriteLine(" [" + i + "]  " + weights[i].ToString("F4"));
            Console.WriteLine(" [" + (weights.Length - 1) + "] (bias) " +
              weights[weights.Length - 1].ToString("F4"));
            Console.WriteLine("");


            //check train data itself~~~~~~notice: high accuracy but not alway 100% right
            int total = ptsTrain.Count;
            int correct = 0;
            for (int i = 0; i < ptsTrain.Count; i++)
                if (Accuracy(ptsTrain[i], ptsTrain, sigma, true, weights))
                    correct++;
            Console.WriteLine("\nTrain Data Accurate rate: " + ((double)correct / (double)total) + "\n");



            //set all canvas to be test points
            for (int Y = 0; Y < HEIGHT; Y++)
                for (int X = 0; X < WIDTH; X++)
                    ptsTest.Add(new SamplePoint(X, Y, 0));//0 is color, since this is for ptsTest, color is useless


            for (int X = 0; X < WIDTH; X++)
                for (int Y = 0; Y < HEIGHT; Y++)
                    ptsTest.Add(new SamplePoint(X, Y, 0));//0 is color, since this is for ptsTest, color is useless


            //draw seperate line
            bool change = Accuracy(ptsTest[0], ptsTrain, sigma, false, weights);
            for (int i = 1; i < ptsTest.Count; i++)
            {
                if (change != Accuracy(ptsTest[i], ptsTrain, sigma, false, weights))
                {
                    change = !change;
                    Canvas_Draw_Point(ptsTest[i].nX, ptsTest[i].nY, true, 2, 3, true);
                }
            }
        }

        bool Accuracy(SamplePoint inputData, List<SamplePoint> trainData, double sigma, bool isTesting, double[] weights)
        {
            int numTrain = trainData.Count;

            // compare currrent against all trainData
            double i_sum = 0.0;
            for (int j = 0; j < weights.Length - 1; ++j)
            {
                double k = Kernel(inputData, trainData[j], sigma);  //依序跟每個train sample計算RBF值     較小的 RBF 值指出兩個點比較不類似(距離遠)
                i_sum += weights[j] * k;  // (cannot pre-compute) 
            }
            i_sum += weights[weights.Length - 1] * 1;  // add the bias

            double y = 1.0 / (1.0 + Math.Exp(-i_sum));
            double color = inputData.Color;

            if (isTesting)
            {
                if (y <= 0.5 && color == 0.0 || y > 0.5 && color == 1.0)
                    return true;//test result ok
                else
                    return false;//test result fail
            }
            else // not verbose
            {
                if (y <= 0.5)
                    return true;//color = 0
                else
                    return false;//color = 1
            }
        }

        static double Kernel(SamplePoint v1, SamplePoint v2, double sigma)
        {
            // RBF kernel. v1 & v2 have class label in last cell
            double num = 0.0;

            num += Math.Pow((v1.dX - v2.dX), 2);
            num += Math.Pow((v1.dY - v2.dY), 2);

            double denom = 2.0 * sigma * sigma;
            double z = num / denom; //RBF: （高斯）徑向基函數核（英語：Radial basis function kernel），或稱為RBF核，是一種常用的核函數
            return Math.Exp(-z);
        }

        static void Shuffle(int[] indices)
        {
            // assumes class-scope Random object rnd
            for (int i = 0; i < indices.Length; ++i)
            {
                int ri = rnd.Next(i, indices.Length);
                int tmp = indices[i];
                indices[i] = indices[ri];
                indices[ri] = tmp;
            }
        }
    }
}
