import 'package:flutter/material.dart';
import 'data.dart';
import 'prediction_page.dart';
import 'prediction_chart.dart';

List<StockSeries> chartData = [];
List<DataRow> tableData = [];

void main() {
  setDataChart().then((cData) {
    setDataTable().then((tData) {
      chartData = cData;
      tableData = tData;
      runApp(MyApp());
    });
  });
}

class MyApp extends StatelessWidget {
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: PredictionPage(), //This is where we specify our homepage
    );
  }
}
