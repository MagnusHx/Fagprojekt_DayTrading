from kvant.ml_framework.train.metric_registry import dashboard_contract_rows, metric_inventory_rows


def test_metric_inventory_tracks_current_side_meta_pipeline() -> None:
    names = {row["name"] for row in metric_inventory_rows()}

    assert "{split}/meta/f1" in names
    assert "{split}/classification/f1_macro" in names
    assert "{split}/portfolio/cumulative_annual_profit_pct" in names
    assert "{split}/portfolio/n_executed_trades" in names
    assert "{split}/paper/executed_trade_net_return_total_pct" in names
    assert "perf/confusion_matrix_normalized/{split}" in names

    assert "{split}/decision/meta_accept_threshold" not in names
    assert "{split}/portfolio/final_balance" not in names
    assert "{split}/paper/tp" not in names


def test_dashboard_contract_uses_current_pipeline_metrics() -> None:
    metrics = {row["metric"] for row in dashboard_contract_rows()}

    assert "{split}/meta/f1" in metrics
    assert "{split}/classification/f1_macro" in metrics
    assert "{split}/portfolio/cumulative_annual_profit_pct" in metrics
    assert "{split}/portfolio/n_executed_trades" in metrics

    assert "{split}/decision/meta_accept_threshold" not in metrics
