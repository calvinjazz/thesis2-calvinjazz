import 'package:flutter/material.dart';
import 'main.dart';

DataTable predictionTable = _createDataTable();

DataTable _createDataTable() {
  return DataTable(columns: _createColumns(), rows: _createRows());
}

List<DataColumn> _createColumns() {
  return [
    const DataColumn(label: Center(child: Text('Date'))),
    const DataColumn(label: Center(child: Text('Open'))),
    const DataColumn(label: Center(child: Text('High'))),
    const DataColumn(label: Center(child: Text('Low'))),
    const DataColumn(label: Center(child: Text('Close'))),
    const DataColumn(label: Center(child: Text('PSEi Returns'))),
    const DataColumn(label: Center(child: Text('Strategy Returns'))),
    const DataColumn(label: Center(child: Text('Recommended Action'))),
  ];
}

List<DataRow> _createRows() {
  return tableData;
}
