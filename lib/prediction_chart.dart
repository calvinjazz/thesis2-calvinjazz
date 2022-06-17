import 'package:flutter/material.dart';
import 'main.dart';
import 'package:charts_flutter/flutter.dart' as charts;

import 'dart:html';
import 'package:intl/intl.dart';

import 'package:charts_flutter/src/text_element.dart' as charts_text;
import 'package:charts_flutter/src/text_element.dart';
import 'package:charts_flutter/src/text_style.dart' as style;

class PredictionChart extends StatelessWidget {
  PredictionChart({data});

  @override
  Widget build(BuildContext context) {
    List<charts.Series<StockSeries, DateTime>> series = [
      charts.Series(
          id: "pSEiReturns",
          data: chartData,
          domainFn: (StockSeries series, _) => series.date,
          measureFn: (StockSeries series, _) => series.pSEiReturns,
          colorFn: (StockSeries series, _) => series.pSEiReturnsColor),
      charts.Series(
          id: "strategyReturns",
          data: chartData,
          domainFn: (StockSeries series, _) => series.date,
          measureFn: (StockSeries series, _) => series.strategyReturns,
          colorFn: (StockSeries series, _) => series.strategyReturnsColor)
    ];

    return Container(
      height: 500,
      padding: const EdgeInsets.all(25),
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(9.0),
          child: Column(children: <Widget>[
            Text(
              "Chart",
              style: Theme.of(context).textTheme.bodyText2,
            ),
            Expanded(
                child: charts.TimeSeriesChart(
              series,
              animate: true,
              defaultRenderer: charts.LineRendererConfig(),
              dateTimeFactory: const charts.LocalDateTimeFactory(),
              behaviors: [
                charts.SeriesLegend(),
                charts.SlidingViewport(),
                charts.PanAndZoomBehavior(),
                charts.LinePointHighlighter(
                  symbolRenderer: CustomCircleSymbolRenderer(),
                )
              ],
              selectionModels: [
                charts.SelectionModelConfig(
                    changedListener: (charts.SelectionModel model) {
                  if (model.hasDatumSelection) {
                    final value = model.selectedSeries[0]
                        .measureFn(model.selectedDatum[0].index);
                    CustomCircleSymbolRenderer.value =
                        value.toString(); // paints the tapped value
                  }
                })
              ],
              domainAxis: charts.DateTimeAxisSpec(
                tickProviderSpec: const charts.DayTickProviderSpec(
                    increments: [15]), // View in increments of 30 days
                viewport: charts.DateTimeExtents(
                  start: DateTime(
                      chartData.last.date.year,
                      chartData.last.date.month - 3,
                      chartData.last.date
                          .day), // Start Date 6 months before current date
                  end: DateTime(chartData.last.date.year,
                      chartData.last.date.month + 1, chartData.last.date.day),
                ),
              ),
              primaryMeasureAxis: charts.NumericAxisSpec(
                  viewport: const charts.NumericExtents(-30, 30),
                  tickFormatterSpec:
                      charts.BasicNumericTickFormatterSpec.fromNumberFormat(
                          NumberFormat("#.##")),
                  tickProviderSpec: const charts.BasicNumericTickProviderSpec(
                      zeroBound: false, dataIsInWholeNumbers: false)),
            ))
          ]),
        ),
      ),
    );
  }
}

class StockSeries {
  final DateTime date;
  final double pSEiReturns;
  final double strategyReturns;
  final charts.Color pSEiReturnsColor;
  final charts.Color strategyReturnsColor;

  StockSeries(
      {required this.date,
      required this.pSEiReturns,
      required this.strategyReturns,
      required this.pSEiReturnsColor,
      required this.strategyReturnsColor});
}

class CustomCircleSymbolRenderer extends charts.CircleSymbolRenderer {
  static String? value;
  @override
  void paint(charts.ChartCanvas canvas, Rectangle<num> bounds,
      {List<int>? dashPattern,
      charts.Color? fillColor,
      charts.FillPatternType? fillPattern,
      charts.Color? strokeColor,
      double? strokeWidthPx}) {
    super.paint(canvas, bounds,
        dashPattern: dashPattern,
        fillColor: fillColor,
        fillPattern: fillPattern,
        strokeColor: strokeColor,
        strokeWidthPx: strokeWidthPx);
    canvas.drawRect(
        Rectangle(bounds.left - 5, bounds.top - 30, bounds.width + 10,
            bounds.height + 10),
        fill: charts.Color.white);
    var textStyle = style.TextStyle();
    textStyle.color = charts.Color.black;
    textStyle.fontSize = 15;
    canvas.drawText(charts_text.TextElement("$value", style: textStyle),
        (bounds.left).round(), (bounds.top - 28).round());
  }
}
