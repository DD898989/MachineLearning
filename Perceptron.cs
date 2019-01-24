using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Runtime.InteropServices;

namespace WindowsFormsApplication24
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
        double w0, w1, w2;   // 0 = w0*x + w1*y + w2
        const double X0 = -1;
        Graphics objGraphics;
        List<SamplePoint> pts = new List<SamplePoint>();


        [DllImport("kernel32.dll", SetLastError = true)] //AllocConsole();
        [return: MarshalAs(UnmanagedType.Bool)]          //AllocConsole();
        static extern bool AllocConsole(); //AllocConsole();

        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.NumericUpDown numericUpDown1;
        private System.Windows.Forms.Panel panel1;
        public Form1()
        {
            AllocConsole();

            InitializeComponent();
            //-------------------------------------------------------------
            this.button1 = new System.Windows.Forms.Button();
            this.button1.Location = new System.Drawing.Point(13, 13);
            this.button2 = new System.Windows.Forms.Button();
            this.button2.Location = new System.Drawing.Point(13, 42);
            this.numericUpDown1 = new System.Windows.Forms.NumericUpDown();
            this.numericUpDown1.Location = new System.Drawing.Point(13, 72);
            this.panel1 = new System.Windows.Forms.Panel();
            this.panel1.Location = new System.Drawing.Point(13, 101);
            // Form1
            this.ClientSize = new System.Drawing.Size(524, 424);
            this.Controls.Add(this.panel1);
            this.Controls.Add(this.numericUpDown1);
            this.Controls.Add(this.button2);
            this.Controls.Add(this.button1);
            //-------------------------------------------------------------
            this.button1.Click += new System.EventHandler(this.btnLearn_Click);
            this.button1.Text = "Learn";
            this.button2.Click += new System.EventHandler(this.btnClearAll_Click);
            this.button2.Text = "Clear";
            this.panel1.MouseDown += new System.Windows.Forms.MouseEventHandler(this.Canvas_MouseDown);
            this.panel1.Width = this.Width;
            this.panel1.Height = this.Height;
            this.panel1.Location = new System.Drawing.Point(0, 0);
            this.panel1.SendToBack();
            this.numericUpDown1.DecimalPlaces = 1;
            this.numericUpDown1.Value = 0M;
            this.numericUpDown1.Increment = 0.1M;
            this.numericUpDown1.Minimum = 0M;
            this.numericUpDown1.Maximum = 1M;
            //-------------------------------------------------------------

            objGraphics = panel1.CreateGraphics();
            WIDTH = panel1.Width;
            HEIGHT = panel1.Height;
        }


        private void DrawPoint(int X, int Y, bool bColor, float penWid = 2, int penLen = 3)
        {
            Pen pen = new Pen(Color.Red);
            pen.Width = penWid;
            if (bColor)
                pen.Color = Color.Blue;

            objGraphics.DrawLine(pen, new Point(X - penLen, Y), new Point(X + penLen, Y));
            objGraphics.DrawLine(pen, new Point(X, Y - penLen), new Point(X, Y + penLen));
        }

        private void Canvas_MouseDown(object sender, MouseEventArgs e)
        {
            SamplePoint sample;

            if (e.Button == MouseButtons.Left)
                sample = new SamplePoint(e.X, e.Y, 1);
            else
                sample = new SamplePoint(e.X, e.Y, 0);

            pts.Add(sample);

            DrawPoint(e.X, e.Y, e.Button == MouseButtons.Left);
        }

        private int Classify(double x, double y)
        {
            return (x * w0 + y * w1 + w2 >= 0) ? 1 : 0;
        }

        private int SeperatePointY(int nX)
        {
            double dX = (double)(nX - WIDTH / 2) / 10;
            double dY = (-dX * w0 - w2) / w1;
            int nY = (int)(HEIGHT / 2 - dY * 10);
            return nY;
        }

        private void UpdateCanvas()
        {
            objGraphics.Clear(Color.White);

            Point p1 = new Point(0, SeperatePointY(0));
            Point p2 = new Point(WIDTH, SeperatePointY(WIDTH));

            if (w2 != 0)
            {
                //seperate line
                objGraphics.DrawLine(new Pen(Color.DarkGreen), p1, p2);

                //origin
                foreach (SamplePoint pt in pts)
                    DrawPoint(pt.nX, pt.nY, pt.Color == 1, 2, 10);

                //classified
                foreach (SamplePoint pt in pts)
                {
                    bool bColor = Convert.ToBoolean(Classify(pt.dX, pt.dY));
                    DrawPoint(pt.nX, pt.nY, bColor, 5);
                }
            }
        }

        private void btnClearAll_Click(object sender, EventArgs e)
        {
            pts.Clear();
            objGraphics.Clear(Color.White);
        }

        private void btnLearn_Click(object sender, EventArgs e)
        {
            Random rnd = new Random();
            w0 = rnd.NextDouble();
            w1 = rnd.NextDouble();
            w2 = rnd.NextDouble();
            //for 2D line : return (x * w0 + y * w1 + w2 >= 0) ? 1 : 0;
            //for 3D plain: return (x * w0 + y * w1 + z*w3 + w4 >= 0) ? 1 : 0;

            bool error = true;
            int iterations = 0;
            int maxIterations = 1000;// int.Parse(textBox_NumberOfIterations.Text);
            double learningRate = Convert.ToDouble(numericUpDown1.Value);  //if =0,   w0,w1,w1 = initial random value

            while (error)
            {
                if (iterations > maxIterations)//couldn't find perfect seperate line
                    break;

                error = false;

                for (int i = 0; i <= pts.Count - 1; i++)
                {
                    double x1 = pts[i].dX;
                    double x2 = pts[i].dY;

                    int color = Classify(x1, x2);
                    int updateDirection = (pts[i].Color - color);//if  = 0: correct, dont update

                    if (Convert.ToBoolean(updateDirection))
                        error = true;

                    w0 += learningRate * updateDirection * x1;
                    w1 += learningRate * updateDirection * x2;
                    w2 += learningRate * updateDirection * 1;
                }
                iterations++;
            }
            UpdateCanvas();
        }
    }
}
