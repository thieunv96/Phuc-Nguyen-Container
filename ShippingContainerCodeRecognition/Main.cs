using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.OCR;
using Emgu.CV.ML;
using System.Reflection;
using System.IO;

namespace ShippingContainerCodeRecognition
{
    public partial class Main : Form
    {
        Image<Bgr, byte> mImgInput;
        Image<Bgr, byte> mImgDetected;
        Image<Bgr, byte> mImgCroped;
        Image<Gray, byte> mImgSegment;
        Image<Bgr, byte> mImgCharSegment;
        Image<Bgr, byte> mImgCharBox;
        
        public Main()
        {
            InitializeComponent();
        }

        private void Main_Load(object sender, EventArgs e)
        {

        }

        private void btBrowser_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog ofd = new OpenFileDialog())
            {
                ofd.Filter = "Image file | *.png;*.jpg;*.bmp;*.jpeg;*.tiff";
                if(ofd.ShowDialog() == DialogResult.OK)
                {
                    if(mImgInput != null)
                    {
                        mImgInput.Dispose();
                        mImgInput = null;
                    }
                    mImgInput = new Image<Bgr, byte>(ofd.FileName);
                    Processing(mImgInput);
                    imb1.Image = mImgInput.Bitmap;
                }
            }
        }
        private void Processing(Image<Bgr, byte> ImgSource, TextColor Color = TextColor.White)
        {
            Rectangle ROICode = new Rectangle();
            mImgDetected = ImgSource.Copy();
            // create and ROI image
            Rectangle ROI = new Rectangle(ImgSource.Width / 2, ImgSource.Height / 10, ImgSource.Width, ImgSource.Height/4);
            mImgDetected.ROI = ROI;
            // filter noise
            //detect code
            using (Image<Gray, byte> imgGray = mImgDetected.Convert<Gray, byte>())
            {
                using (Image < Gray, byte> imgFilter = new Image<Gray, byte>(imgGray.Size))
                {
                    CvInvoke.BilateralFilter(imgGray, imgFilter, 9, 49, 49);
                    using (Mat k = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, new Size(5, 5), new Point(-1, -1)))
                    {
                        if(Color == TextColor.White)
                            CvInvoke.Erode(imgFilter, imgFilter, k, new Point(-1, -1), 1, Emgu.CV.CvEnum.BorderType.Default, new MCvScalar());
                        else
                            CvInvoke.Dilate(imgFilter, imgFilter, k, new Point(-1, -1), 1, Emgu.CV.CvEnum.BorderType.Default, new MCvScalar());
                    }
                    using (Image<Gray, double> ImgSobel = new Image<Gray, double>(imgFilter.Size))
                    {
                        CvInvoke.Sobel(imgFilter, ImgSobel, Emgu.CV.CvEnum.DepthType.Cv64F, 1, 0, kSize:1);
                        CvInvoke.ConvertScaleAbs(ImgSobel, imgFilter, 2 , 0);
                        CvInvoke.Threshold(imgFilter, imgFilter, 20, 255, Emgu.CV.CvEnum.ThresholdType.Binary);
                            
                        using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
                        {
                            CvInvoke.FindContours(imgFilter, contours, null, Emgu.CV.CvEnum.RetrType.External, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);
                            for (int i = 0; i < contours.Size; i++)
                            {
                                double s = CvInvoke.ContourArea(contours[i]);
                                Rectangle bound = CvInvoke.BoundingRectangle(contours[i]);
                                if(bound.Height > 65 || s < 10)
                                {
                                    CvInvoke.DrawContours(imgFilter, contours, i, new MCvScalar(0), -1);
                                }
                            }
                        }
                        using (Mat k = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, new Size(107, 1), new Point(-1, -1)))
                        {
                             CvInvoke.MorphologyEx(imgFilter, imgFilter, Emgu.CV.CvEnum.MorphOp.Close, k, new Point(-1, -1), 1, Emgu.CV.CvEnum.BorderType.Default, new MCvScalar());
                        }
                        using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
                        {
                            CvInvoke.FindContours(imgFilter, contours, null, Emgu.CV.CvEnum.RetrType.External, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);
                            double large_area = 0;
                            int index_large = 0;
                            for (int i = 0; i < contours.Size; i++)
                            {
                                double s = CvInvoke.ContourArea(contours[i]);
                                if(large_area < s)
                                {
                                    large_area = s;
                                    index_large = i;
                                }
                            }
                            Rectangle boxFirstLine = CvInvoke.BoundingRectangle(contours[index_large]);
                            Rectangle boxSecondLine = new Rectangle();
                            for (int i = 0; i < contours.Size; i++)
                            {
                                Rectangle b = CvInvoke.BoundingRectangle(contours[i]);
                                if(b.Y - boxFirstLine.Y < 120 && b.Y - boxFirstLine.Y > 0 && b.Width > 30)
                                {
                                    boxSecondLine = CvInvoke.BoundingRectangle(contours[i]);
                                    break;
                                }
                            }
                            ROICode = new Rectangle(boxFirstLine.X -20, boxFirstLine.Y - 20, boxFirstLine.Width + 40, boxSecondLine.Y + boxSecondLine.Height + 80 - boxFirstLine.X);
                            ROICode.X = ROICode.X < 0 ? 0: ROICode.X;
                            ROICode.Y = ROICode.Y < 0 ? 0 : ROICode.Y;
                            ROICode.Width = ROICode.X + ROICode.Width > mImgDetected.Width ? mImgDetected.Width - ROICode.X : ROICode.Width;
                            ROICode.Height = ROICode.Y + ROICode.Height > mImgDetected.Height ? mImgDetected.Height - ROICode.Y : ROICode.Height;
                            mImgCroped = mImgDetected.Copy();
                            
                            mImgCroped.ROI = ROICode;
                            using (Image<Bgr, byte> temp = mImgCroped.Copy())
                            {
                                CvInvoke.BilateralFilter(temp, mImgCroped, 9, 49, 49);
                            }
                            CvInvoke.Rectangle(mImgDetected, ROICode, new MCvScalar(255, 0, 0), 3);
                            mImgDetected.ROI = new Rectangle();
                            imb3.Image = mImgCroped.Bitmap;

                        }
                    }
                    
                }
            }
            // segment char text
            mImgSegment = new Image<Gray, byte>(mImgCroped.Size);
            mImgCharSegment = mImgCroped.Copy();
            mImgCharBox = mImgCroped.Copy();
            CvInvoke.CvtColor(mImgCroped, mImgSegment, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);
            using (Mat k = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, new Size(5, 5), new Point(-1, -1)))
            {
                CvInvoke.MorphologyEx(mImgSegment, mImgSegment, Emgu.CV.CvEnum.MorphOp.Open, k, new Point(-1, -1), 1, Emgu.CV.CvEnum.BorderType.Default, new MCvScalar());
            }
            CvInvoke.Imwrite("test.png", mImgSegment);
            CvInvoke.Threshold(mImgSegment, mImgSegment, 127, 255, Emgu.CV.CvEnum.ThresholdType.Binary);
            //CvInvoke.Imwrite("test.png", mImgSegment);
            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                CvInvoke.FindContours(mImgSegment, contours, null, Emgu.CV.CvEnum.RetrType.External, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);
                for (int i = 0; i < contours.Size; i++)
                {
                    Rectangle bound = CvInvoke.BoundingRectangle(contours[i]);
                    if (bound.Height > 60 || bound.Height < 30 || bound.Width > 35)
                    {
                        CvInvoke.DrawContours(mImgSegment, contours, i, new MCvScalar(0), -1);
                    }
                }
            }
            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                CvInvoke.FindContours(mImgSegment, contours, null, Emgu.CV.CvEnum.RetrType.External, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);
                for (int i = 0; i < contours.Size; i++)
                {
                    Rectangle bound = CvInvoke.BoundingRectangle(contours[i]);
                    CvInvoke.Rectangle(mImgCharSegment, bound, new MCvScalar(0, 255, 0), 1);
                }
            }
            imb5.Image = mImgCharSegment.Bitmap;
            imb4.Image = mImgSegment.Copy().Bitmap;
            CvInvoke.Threshold(mImgSegment, mImgSegment, 127, 255, Emgu.CV.CvEnum.ThresholdType.BinaryInv);
            string code = Read(mImgSegment);
            CvInvoke.PutText(mImgCharBox, code, new Point(10, 22), Emgu.CV.CvEnum.FontFace.HersheySimplex, 1, new MCvScalar(255, 0, 0), 2);
            txtCode.Text = code;
            txtCode.ForeColor = System.Drawing.Color.Violet;
            imb6.Image = mImgCharBox.Bitmap;
            imb2.Image = mImgDetected.Bitmap;
        }
        private string Read(Image<Gray, byte> Img)
        {
            Image<Gray, byte> ImgResize = Img.Resize(Img.Width, Convert.ToInt32(0.6 * Img.Height), Emgu.CV.CvEnum.Inter.Linear);
            string str = string.Empty;
            string path = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) + "\\";
            using (var _ocr = new Tesseract(path, "eng", OcrEngineMode.Default))
            {
                _ocr.SetImage(ImgResize);
                _ocr.Recognize();
                string s = _ocr.GetBoxText(0);
                s = s.Replace("\r", "");
                string[] eachChar = s.Split('\n');
                
                List<Rectangle> box = new List<Rectangle>();
                for (int i = 0; i < eachChar.Length; i++)
                {
                    if (eachChar[i] == "")
                        continue;
                    string[] chr = eachChar[i].Split(' ');

                    char ct = Convert.ToChar(chr[0]);
                    int val = (int)ct;
                    if ((val >= 0x30 && val <= 0x39) || (val >= 0x41 && val <= 0x5a) || (val >= 0x61 && val <= 0x7a))
                    {
                        str += chr[0];
                        box.Add(new Rectangle(Convert.ToInt32(chr[1]), Convert.ToInt32(chr[2]), Convert.ToInt32(chr[3]) - Convert.ToInt32(chr[1]), Convert.ToInt32(chr[4]) - Convert.ToInt32(chr[2])));
                    }
                }
            }
            return str;
        }


    }
    enum TextColor
    {
        Black,
        White
    }
}
