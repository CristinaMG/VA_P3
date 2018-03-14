#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    cap = new VideoCapture(0);
    if(!cap->isOpened())
        cap = new VideoCapture(1);
    capture = true;
    showColorImage = false;
    winSelected = false;
    cap->set(CV_CAP_PROP_FRAME_WIDTH, 320);
    cap->set(CV_CAP_PROP_FRAME_HEIGHT, 240);
    imgS = new QImage(320,240, QImage::Format_RGB888);
    visorS = new RCDraw(320,240, imgS, ui->imageFrameS);
    imgD = new QImage(320,240, QImage::Format_RGB888);
    visorD = new RCDraw(320,240, imgD, ui->imageFrameD);

    colorImage.create(240,320,CV_8UC3);
    grayImage.create(240,320,CV_8UC1);
    destColorImage.create(240,320,CV_8UC3);
    destGrayImage.create(240,320,CV_8UC1);
    gray2ColorImage.create(240,320,CV_8UC3);
    destGray2ColorImage.create(240,320,CV_8UC3);

    matcherObj = BFMatcher();

    connect(&timer,SIGNAL(timeout()),this,SLOT(compute()));
    connect(ui->captureButton,SIGNAL(clicked(bool)),this,SLOT(start_stop_capture(bool)));
    connect(ui->colorButton,SIGNAL(clicked(bool)),this,SLOT(change_color_gray(bool)));
    connect(visorS,SIGNAL(windowSelected(QPointF, int, int)),this,SLOT(selectWindow(QPointF, int, int)));
    connect(visorS,SIGNAL(pressEvent()),this,SLOT(deselectWindow()));

    connect(ui->addButton, SIGNAL(clicked(bool)), this, SLOT(extract_descriptor()));

    timer.start(60);


}

MainWindow::~MainWindow()
{
    delete ui;
    delete cap;
    delete visorS;
    delete visorD;
    delete imgS;
    delete imgD;

}

void MainWindow::compute()
{
    if(capture && cap->isOpened())
    {
        *cap >> colorImage;

        cvtColor(colorImage, grayImage, CV_BGR2GRAY);
        cvtColor(colorImage, colorImage, CV_BGR2RGB);
    }

    update_image();

    if(showColorImage)
    {
        memcpy(imgS->bits(), colorImage.data , 320*240*3*sizeof(uchar));
        memcpy(imgD->bits(), destColorImage.data , 320*240*3*sizeof(uchar));
    }
    else
    {
        cvtColor(grayImage,gray2ColorImage, CV_GRAY2RGB);
        cvtColor(destGrayImage,destGray2ColorImage, CV_GRAY2RGB);
        memcpy(imgS->bits(), gray2ColorImage.data , 320*240*3*sizeof(uchar));
        memcpy(imgD->bits(), destGray2ColorImage.data , 320*240*3*sizeof(uchar));
    }

    if(winSelected)
    {
        visorS->drawSquare(QPointF(imageWindow.x+imageWindow.width/2, imageWindow.y+imageWindow.height/2), imageWindow.width,imageWindow.height, Qt::green );
    }
    visorS->update();
    visorD->update();

}

void MainWindow::start_stop_capture(bool start)
{
    if(start)
    {
        ui->captureButton->setText("Stop capture");
        capture = true;
    }
    else
    {
        ui->captureButton->setText("Start capture");
        capture = false;
    }
}

void MainWindow::change_color_gray(bool color)
{
    if(color)
    {
        ui->colorButton->setText("Gray image");
        showColorImage = true;
    }
    else
    {
        ui->colorButton->setText("Color image");
        showColorImage = false;
    }
}

void MainWindow::selectWindow(QPointF p, int w, int h)
{
    QPointF pEnd;
    if(w>0 && h>0)
    {
        imageWindow.x = p.x()-w/2;
        if(imageWindow.x<0)
            imageWindow.x = 0;
        imageWindow.y = p.y()-h/2;
        if(imageWindow.y<0)
            imageWindow.y = 0;
        pEnd.setX(p.x()+w/2);
        if(pEnd.x()>=320)
            pEnd.setX(319);
        pEnd.setY(p.y()+h/2);
        if(pEnd.y()>=240)
            pEnd.setY(239);
        imageWindow.width = pEnd.x()-imageWindow.x;
        imageWindow.height = pEnd.y()-imageWindow.y;

        winSelected = true;
    }
}

void MainWindow::deselectWindow()
{
    winSelected = false;
}


void MainWindow::copyWinSelect(){
    if(winSelected){
           int x = (320-imageWindow.width)/2;
           int y = (240-imageWindow.height)/2;

           destColorImage.setTo(0); //to put in black
           Mat winD = destColorImage(cv::Rect(x, y, imageWindow.width, imageWindow.height));
           Mat(colorImage, imageWindow).copyTo(winD);

           destGrayImage.setTo(0);
           winD = destGrayImage(cv::Rect(x, y, imageWindow.width, imageWindow.height));
           Mat(grayImage, imageWindow).copyTo(winD);
    }
}

void MainWindow::extract_descriptor(){
    //copyWinSelect();

    std::vector<KeyPoint> keyPoints;
    Mat descriptors;
    Ptr<ORB> detector = ORB::create();

    Mat win;
    win.create(imageWindow.height,imageWindow.width,CV_8UC1);
    Mat(grayImage, imageWindow).copyTo(win);
    //tenemos que pasarle exclusivamente el recuadro sin los negros alrededor.
    detector->detectAndCompute(win, Mat(), keyPoints, descriptors,false);

    // 1 bfmatcher y 3 vectores y tienen que ser de la clase.
    // para tener ordenado el bf matcher si se borra o se añade algo se coge y se vuelve a crear de nuevo la colección y meter los 3 vectores en el bfmatcher .
    std::vector<Mat> obj = {win,descriptors};
    switch (ui->comboBox->currentIndex()) {
    case 0:
        arrayDescriptor1.push_back(obj); //push_back añade al final    // para quitarlo pop_back(); creo
        ui->slider->setMaximum(arrayDescriptor1.size()-1);
        ui->slider->setSliderPosition(arrayDescriptor1.size());
        break;
    case 1:
        arrayDescriptor2.push_back(obj);
        ui->slider->setMaximum(arrayDescriptor2.size()-1);
        ui->slider->setSliderPosition(arrayDescriptor2.size());
        break;
    case 2:
        arrayDescriptor3.push_back(obj);
        ui->slider->setMaximum(arrayDescriptor3.size()-1);
        ui->slider->setSliderPosition(arrayDescriptor3.size());
        break;
    default:
        break;
    }

    matcherObj.clear();
    for(auto a: arrayDescriptor1)
        matcherObj.add(a.at(1));

    for(auto a: arrayDescriptor2)
        matcherObj.add(a.at(1));

    for(auto a: arrayDescriptor3)
        matcherObj.add(a.at(1));
}

void MainWindow::update_image(){

    int x, y;
    Mat winD;
    try{
        destGrayImage.setTo(0);
        switch (ui->comboBox->currentIndex()) {
        case 0:
            if(arrayDescriptor1.size()>0){
                ui->slider->setMaximum(arrayDescriptor1.size()-1);
                x = (320-arrayDescriptor1.at(ui->slider->value()).at(0).cols)/2;
                y = (240-arrayDescriptor1.at(ui->slider->value()).at(0).rows)/2;

                winD = destGrayImage(cv::Rect(x, y, arrayDescriptor1.at(ui->slider->value()).at(0).cols, arrayDescriptor1.at(ui->slider->value()).at(0).rows));
                arrayDescriptor1.at(ui->slider->value()).at(0).copyTo(winD);
            }else{
                ui->slider->setSliderPosition(0);
                ui->slider->setMaximum(0);
            }
            break;
        case 1:
            if(arrayDescriptor2.size()>0){
                ui->slider->setMaximum(arrayDescriptor2.size()-1);
                x = (320-arrayDescriptor2.at(ui->slider->value()).at(0).cols)/2;
                y = (240-arrayDescriptor2.at(ui->slider->value()).at(0).rows)/2;

                winD = destGrayImage(cv::Rect(x, y, arrayDescriptor2.at(ui->slider->value()).at(0).cols, arrayDescriptor2.at(ui->slider->value()).at(0).rows));
                arrayDescriptor2.at(ui->slider->value()).at(0).copyTo(winD);
            }else{
                ui->slider->setSliderPosition(0);
                ui->slider->setMaximum(0);
            }
            break;
        case 2:
            if(arrayDescriptor3.size()>0){
                ui->slider->setMaximum(arrayDescriptor3.size()-1);
                x = (320-arrayDescriptor3.at(ui->slider->value()).at(0).cols)/2;
                y = (240-arrayDescriptor3.at(ui->slider->value()).at(0).rows)/2;

                winD = destGrayImage(cv::Rect(x, y, arrayDescriptor3.at(ui->slider->value()).at(0).cols, arrayDescriptor3.at(ui->slider->value()).at(0).rows));
                arrayDescriptor3.at(ui->slider->value()).at(0).copyTo(winD);
            }else{
                ui->slider->setSliderPosition(0);
                ui->slider->setMaximum(0);
            }
            break;
        default:
            break;
        }
    }catch(...){
        qDebug()<<"Error update_image";
    }
}
