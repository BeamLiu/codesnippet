#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

class QLineEdit;
class QPushButton;
class QTextEdit;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);

private slots:
    void handleRun();
    void handleClear();

private:
    QLineEdit *inputLineEdit_;
    QPushButton *runButton_;
    QPushButton *clearButton_;
    QTextEdit *outputTextEdit_;
};

#endif
