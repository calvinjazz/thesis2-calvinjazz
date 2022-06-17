import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'prediction_chart.dart';
import 'package:intl/intl.dart';
import 'package:charts_flutter/flutter.dart' as charts;

const fields = 50;

Future<List> fetchStockData() async {
  final response =
      await http.get(Uri.parse('https://knn-stock-prediction.herokuapp.com/'));

  if (response.statusCode == 200) {
    // If the server did return a 200 OK response,
    // then parse the JSON.
    // return StockData.fromJson(jsonDecode(response.body));
    String arrayText = response.body;
    var tagsJson = jsonDecode(arrayText)['data'];
    List data = List.from(tagsJson);
    return data;
  } else {
    // If the server did not return a 200 OK response,
    // then throw an exception.
    throw Exception('Failed to load data');
  }
}

// Setup for Chart
Future<List<StockSeries>> setDataChart() async {
  List<StockSeries> data = [];
  await fetchStockData().then((stocks) {
    for (var stock
        in stocks.sublist(stocks.length - fields * 4, stocks.length)) {
      final DateTime date = DateTime.parse(stock['Date']);
      final DateFormat formatter = DateFormat('MM-dd-yyyy');
      final String formattedDate = formatter.format(date);

      stock['Cumulative_PSEi_returns'] ??= 0.0; // Set to 0.0 if Null
      stock['Cumulative_Strategy_returns'] ??= 0.0;

      data.add(StockSeries(
        date: date,
        pSEiReturns: stock['Cumulative_PSEi_returns'],
        strategyReturns: stock['Cumulative_Strategy_returns'],
        pSEiReturnsColor: charts.ColorUtil.fromDartColor(Colors.red),
        strategyReturnsColor: charts.ColorUtil.fromDartColor(Colors.green),
      ));
    }
    data = data;
  });
  return data;
}

// Setup for
Future<List<DataRow>> setDataTable() async {
  List<DataRow> data = [];
  await fetchStockData().then((stocks) {
    int loop = fields;

    for (var stock in stocks.reversed) {
      final DateTime date = DateTime.parse(stock['Date']);
      final DateFormat formatter = DateFormat('MM-dd-yyyy');
      final String formattedDate = formatter.format(date);

      stock['PSEi_movement'] ??= 0.0; // Set to 0.0 if Null
      stock['Startegy_movement'] ??= 0.0;

      String action = stock['Predicted_Signal'] == 1 ? "Buy" : "Sell";

      data.add(DataRow(cells: [
        DataCell(Center(child: Text(formattedDate))),
        DataCell(Center(child: Text(stock['Open'].toString()))),
        DataCell(Center(child: Text(stock['High'].toString()))),
        DataCell(Center(child: Text(stock['Low'].toString()))),
        DataCell(Center(child: Text(stock['Close'].toString()))),
        DataCell(Center(
            child: Text(stock['Cumulative_PSEi_returns'].toStringAsFixed(2)))),
        DataCell(Center(
            child:
                Text(stock['Cumulative_Strategy_returns'].toStringAsFixed(2)))),
        DataCell(Center(child: Text(action))),
      ]));

      if (loop - 1 == 0) break;
      loop--;
    }
    data = data;
  });
  // print(data);
  return data;
}
