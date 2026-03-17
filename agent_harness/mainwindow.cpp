#include "mainwindow.h"

#include <QHBoxLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QStatusBar>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QWidget>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      inputLineEdit_(new QLineEdit(this)),
      runButton_(new QPushButton(tr("Run"), this)),
      clearButton_(new QPushButton(tr("Clear"), this)),
      outputTextEdit_(new QTextEdit(this))
{
    auto *centralWidget = new QWidget(this);
    auto *mainLayout = new QVBoxLayout(centralWidget);
    auto *controlsLayout = new QHBoxLayout();

    inputLineEdit_->setPlaceholderText(tr("Enter command or text"));
    outputTextEdit_->setReadOnly(true);

    controlsLayout->addWidget(inputLineEdit_);
    controlsLayout->addWidget(runButton_);
    controlsLayout->addWidget(clearButton_);

    mainLayout->addLayout(controlsLayout);
    mainLayout->addWidget(outputTextEdit_);

    setCentralWidget(centralWidget);
    statusBar()->showMessage(tr("Status: Ready"));
    setWindowTitle(tr("Agent Harness Qt Sample"));
    resize(720, 480);

    connect(runButton_, &QPushButton::clicked, this, &MainWindow::handleRun);
    connect(clearButton_, &QPushButton::clicked, this, &MainWindow::handleClear);
    connect(inputLineEdit_, &QLineEdit::returnPressed, this, &MainWindow::handleRun);
}

void MainWindow::handleRun()
{
    const QString text = inputLineEdit_->text();
    if (text.isEmpty()) {
        statusBar()->showMessage(tr("Status: Input is empty"));
        return;
    }

    outputTextEdit_->append(QStringLiteral("> %1").arg(text));
    inputLineEdit_->clear();
    statusBar()->showMessage(tr("Status: Command echoed"));
}

void MainWindow::handleClear()
{
    outputTextEdit_->clear();
    statusBar()->showMessage(tr("Status: Output cleared"));
}
