import { useEffect, useMemo, useState } from 'react'
import { API_BASE_URL } from './config'
import './ModelInsights.css'

const API_URL = `${API_BASE_URL}/insights/overview`

const metricDefinitions = [
  { key: 'accuracy', label: 'Accuracy', note: 'Overall correct classifications' },
  { key: 'precision', label: 'Precision', note: 'Positive predictions that were correct' },
  { key: 'recall', label: 'Recall / Sensitivity', note: 'Actual positive cases identified' },
  { key: 'specificity', label: 'Specificity', note: 'Actual negative cases identified' },
  { key: 'f1_score', label: 'F1 score', note: 'Balance of precision and recall' },
  { key: 'roc_auc', label: 'ROC-AUC', note: 'Ranking discrimination ability' },
]

function percent(value) {
  return value == null ? '—' : `${(value * 100).toFixed(2)}%`
}

function MetricCard({ label, value, note, accent = 'cyan' }) {
  return (
    <article className={`mi-metric-card ${accent}`}>
      <span className="mi-card-label">{label}</span>
      <strong>{value}</strong>
      <span className="mi-card-note">{note}</span>
    </article>
  )
}

function ModelInsights({ authToken, onLogout }) {
  const [insights, setInsights] = useState(null)
  const [activeTab, setActiveTab] = useState('overview')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    const controller = new AbortController()

    async function loadInsights() {
      setLoading(true)
      setError('')

      try {
        const response = await fetch(API_URL, {
          method: 'GET',
          headers: {
            Authorization: `Bearer ${authToken}`,
            'Content-Type': 'application/json',
          },
          signal: controller.signal,
        })

                // Automatically log out if the session has expired.
        if (response.status === 401) {
          onLogout()
          return
        }

        const payload = await response.json()

        if (!response.ok) {
          throw new Error(payload.detail || 'Unable to load model insights.')
        }

        setInsights(payload)
      } catch (requestError) {
        if (requestError.name !== 'AbortError') {
          setError(requestError.message || 'Something went wrong.')
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false)
      }
    }

    if (authToken) loadInsights()
    else {
      setError('Authentication is required to view Model Insights.')
      setLoading(false)
    }

    return () => controller.abort()
  }, [authToken])

  const strongestModel = useMemo(() => {
    if (!insights?.model?.algorithm_comparison?.length) return null
    return [...insights.model.algorithm_comparison].sort(
      (a, b) => (b.roc_auc ?? 0) - (a.roc_auc ?? 0),
    )[0]
  }, [insights])

  if (loading) {
    return (
      <section className="dashboard-content model-insights-page">
        <div className="mi-state-card">
          <span className="mi-loader" />
          <h2>Preparing model intelligence</h2>
          <p>Reading the dataset and saved research evaluation results.</p>
        </div>
      </section>
    )
  }

  if (error || !insights) {
    return (
      <section className="dashboard-content model-insights-page">
        <div className="mi-state-card mi-error-state">
          <span className="mi-state-icon">!</span>
          <h2>Model insights unavailable</h2>
          <p>{error || 'The backend returned an empty response.'}</p>
          <p className="mi-state-hint">
            Confirm that FastAPI is running and the project data/results folders are present.
          </p>
        </div>
      </section>
    )
  }

  const { project, dataset, model } = insights
  const test = model.independent_test
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'comparison', label: 'Algorithm Comparison' },
    { id: 'dataset', label: 'Dataset Intelligence' },
    { id: 'correlation', label: 'Feature Correlation' },
  ]

  return (
    <section className="dashboard-content model-insights-page">
      <header className="mi-page-heading">
        <div>
          <p className="section-eyebrow">INTELLIGENCE CENTER · RESEARCH ANALYTICS</p>
          <h2>Model Insights</h2>
          <p>
            Explore the evidence behind DiaPredict — model performance,
            dataset characteristics and feature relationships.
          </p>
        </div>
        <div className="mi-live-label"><span /> RESEARCH DATA</div>
      </header>

      <div className="mi-context-strip">
        <div>
          <span>SELECTED MODEL</span>
          <strong>{model.selected_algorithm}</strong>
        </div>
        <div>
          <span>DATASET</span>
          <strong>{project.dataset_name}</strong>
        </div>
        <div>
          <span>RECORDS</span>
          <strong>{project.dataset_records.toLocaleString()}</strong>
        </div>
        <div>
          <span>INPUT FEATURES</span>
          <strong>{project.feature_count}</strong>
        </div>
      </div>

      <nav className="mi-tabs" aria-label="Model insights sections">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={activeTab === tab.id ? 'active' : ''}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {activeTab === 'overview' && (
        <>
          <div className="mi-section-heading">
            <div>
              <p className="section-eyebrow">INDEPENDENT TEST EVALUATION</p>
              <h3>Selected model performance</h3>
              <p>Metrics below are from the saved independent test evaluation.</p>
            </div>
            <span className="mi-threshold-pill">Threshold {test.threshold}</span>
          </div>

          <div className="mi-metric-grid">
            {metricDefinitions.map((metric, index) => (
              <MetricCard
                key={metric.key}
                label={metric.label}
                value={percent(test[metric.key])}
                note={metric.note}
                accent={['cyan', 'blue', 'violet', 'green', 'amber', 'pink'][index]}
              />
            ))}
          </div>

          <div className="mi-two-column">
            <article className="mi-panel">
              <div className="mi-panel-heading">
                <div>
                  <span className="mi-panel-kicker">CLASSIFICATION DETAILS</span>
                  <h3>Confusion matrix</h3>
                </div>
                <span className="mi-panel-tag">N = {test.records}</span>
              </div>
              <div className="mi-confusion-grid">
                <div className="mi-cm-axis">Predicted 0</div>
                <div className="mi-cm-axis">Predicted 1</div>
                <div className="mi-cm-cell true">
                  <strong>{test.confusion_matrix.true_negative}</strong>
                  <span>True Negative · Actual 0</span>
                </div>
                <div className="mi-cm-cell false">
                  <strong>{test.confusion_matrix.false_positive}</strong>
                  <span>False Positive · Actual 0</span>
                </div>
                <div className="mi-cm-cell false">
                  <strong>{test.confusion_matrix.false_negative}</strong>
                  <span>False Negative · Actual 1</span>
                </div>
                <div className="mi-cm-cell true">
                  <strong>{test.confusion_matrix.true_positive}</strong>
                  <span>True Positive · Actual 1</span>
                </div>
              </div>
              <p className="mi-panel-footnote">
                Rows represent actual classes; columns represent predicted classes.
              </p>
            </article>

            <article className="mi-panel">
              <div className="mi-panel-heading">
                <div>
                  <span className="mi-panel-kicker">RESEARCH INTERPRETATION</span>
                  <h3>What these results indicate</h3>
                </div>
              </div>
              <div className="mi-insight-list">
                <div>
                  <span className="mi-insight-dot cyan" />
                  <p><strong>{percent(test.roc_auc)} ROC-AUC</strong> on the independent test set indicates the model's ability to rank positive cases above negative cases.</p>
                </div>
                <div>
                  <span className="mi-insight-dot violet" />
                  <p><strong>{percent(test.recall)} recall</strong> means the model identified this proportion of actual positive cases in the test set.</p>
                </div>
                <div>
                  <span className="mi-insight-dot amber" />
                  <p><strong>{test.confusion_matrix.false_negative} false negatives</strong> occurred at the selected threshold. This matters when interpreting screening-oriented use cases.</p>
                </div>
              </div>
              <p className="mi-caution">
                These are research metrics, not evidence of clinical readiness or a medical diagnosis.
              </p>
            </article>
          </div>
        </>
      )}

      {activeTab === 'comparison' && (
        <>
          <div className="mi-section-heading">
            <div>
              <p className="section-eyebrow">EXPERIMENTAL EVALUATION</p>
              <h3>Algorithm comparison</h3>
              <p>Saved holdout comparison results. Values are not cross-validation scores.</p>
            </div>
          </div>
          <article className="mi-panel mi-table-panel">
            <div className="mi-panel-heading">
              <div>
                <span className="mi-panel-kicker">FIVE CLASSIFIERS</span>
                <h3>Model evaluation metrics</h3>
              </div>
            </div>
            <div className="mi-table-scroll">
              <table className="mi-table">
                <thead>
                  <tr><th>Algorithm</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr>
                </thead>
                <tbody>
                  {model.algorithm_comparison.map((row) => (
                    <tr key={row.model} className={row.model === model.selected_algorithm ? 'selected' : ''}>
                      <td>{row.model}{row.model === model.selected_algorithm && <span className="mi-selected-tag">SELECTED</span>}</td>
                      <td>{percent(row.accuracy)}</td>
                      <td>{percent(row.precision)}</td>
                      <td>{percent(row.recall)}</td>
                      <td>{percent(row.f1_score)}</td>
                      <td>{percent(row.roc_auc)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>

          <div className="mi-two-column mi-comparison-bottom">
            <article className="mi-panel">
              <div className="mi-panel-heading">
                <div><span className="mi-panel-kicker">CROSS-VALIDATION</span><h3>CV ROC-AUC</h3></div>
              </div>
              {model.cross_validation.map((row) => (
                <div className="mi-bar-row" key={row.model}>
                  <div><span>{row.model}</span><strong>{percent(row.roc_auc)}</strong></div>
                  <div className="mi-bar-track"><span style={{ width: `${Math.max(0, Math.min(100, row.roc_auc * 100))}%` }} /></div>
                </div>
              ))}
              <p className="mi-panel-footnote">Cross-validation results are a separate experiment from the independent test results.</p>
            </article>
            <article className="mi-panel">
              <div className="mi-panel-heading">
                <div><span className="mi-panel-kicker">PREPROCESSING EXPERIMENT</span><h3>Pipeline comparison</h3></div>
              </div>
              {model.preprocessing_comparison.map((row) => (
                <div className="mi-preprocess-row" key={row.experiment}>
                  <div><strong>{row.experiment}</strong><span>Accuracy {percent(row.accuracy_mean)}</span></div>
                  <small>F1 {percent(row.f1_mean)} · AUC {percent(row.roc_auc_mean)}</small>
                </div>
              ))}
            </article>
          </div>
          {strongestModel && (
            <p className="mi-analysis-note">
              Highest holdout ROC-AUC in the supplied model comparison: <strong>{strongestModel.model}</strong> ({percent(strongestModel.roc_auc)}).
              This does not automatically determine the selected model; model selection should consider the full evaluation objective.
            </p>
          )}
        </>
      )}

      {activeTab === 'dataset' && (
        <>
          <div className="mi-section-heading">
            <div>
              <p className="section-eyebrow">DATA QUALITY & DISTRIBUTION</p>
              <h3>Dataset intelligence</h3>
              <p>Descriptive statistics calculated from the uploaded CSV.</p>
            </div>
          </div>
          <div className="mi-two-column">
            <article className="mi-panel">
              <div className="mi-panel-heading"><div><span className="mi-panel-kicker">TARGET DISTRIBUTION</span><h3>Outcome classes</h3></div></div>
              {dataset.outcome_distribution.map((item) => (
                <div className="mi-outcome-row" key={item.class}>
                  <div><span>{item.label}</span><strong>{item.count} · {item.percentage}%</strong></div>
                  <div className="mi-bar-track"><span style={{ width: `${item.percentage}%` }} className={item.class === 1 ? 'positive' : ''} /></div>
                </div>
              ))}
              <p className="mi-panel-footnote">The target classes are not evenly distributed; class balance should be considered during evaluation.</p>
            </article>
            <article className="mi-panel">
              <div className="mi-panel-heading"><div><span className="mi-panel-kicker">DATA QUALITY</span><h3>Missing measurement checks</h3></div></div>
              <div className="mi-quality-stat"><strong>{dataset.explicit_missing_cells}</strong><span>Explicit empty cells in source CSV</span></div>
              <div className="mi-zero-list">
                {Object.entries(dataset.zero_as_missing_counts).map(([feature, count]) => (
                  <div key={feature}><span>{feature}</span><strong>{count} zeros</strong></div>
                ))}
              </div>
              <p className="mi-panel-footnote">Zero is treated as a missing measurement for these five variables during descriptive analysis.</p>
            </article>
          </div>
          <article className="mi-panel mi-table-panel">
            <div className="mi-panel-heading"><div><span className="mi-panel-kicker">EIGHT INPUT FEATURES</span><h3>Descriptive statistics</h3></div></div>
            <div className="mi-table-scroll">
              <table className="mi-table mi-feature-table">
                <thead><tr><th>Feature</th><th>Valid n</th><th>Mean</th><th>Median</th><th>Std. dev.</th><th>Min</th><th>Q1</th><th>Q3</th><th>Max</th></tr></thead>
                <tbody>{dataset.feature_summary.map((row) => (
                  <tr key={row.feature}><td>{row.feature}</td><td>{row.count}</td><td>{row.mean}</td><td>{row.median}</td><td>{row.std}</td><td>{row.min}</td><td>{row.q1}</td><td>{row.q3}</td><td>{row.max}</td></tr>
                ))}</tbody>
              </table>
            </div>
          </article>
        </>
      )}

      {activeTab === 'correlation' && (
        <>
          <div className="mi-section-heading">
            <div>
              <p className="section-eyebrow">FEATURE RELATIONSHIPS</p>
              <h3>Pearson correlation matrix</h3>
              <p>Pairwise correlations between the eight input features.</p>
            </div>
          </div>
          <article className="mi-panel mi-correlation-panel">
            <div className="mi-correlation-legend">
              <span><i className="negative" /> Negative</span>
              <span><i className="neutral" /> Near zero</span>
              <span><i className="positive" /> Positive</span>
            </div>
            <div className="mi-heatmap-wrap">
              <div className="mi-heatmap">
                <div className="mi-heatmap-corner" />
                {dataset.correlation_matrix.map((row) => (
                  <div className="mi-heatmap-label" key={`top-${row.feature}`}>{row.feature}</div>
                ))}
                {dataset.correlation_matrix.map((row) => (
                  <div className="mi-heatmap-row" key={row.feature}>
                    <div className="mi-heatmap-side-label">{row.feature}</div>
                    {row.values.map((value, index) => {
                      const safeValue = value ?? 0
                      const intensity = Math.abs(safeValue)
                      const background = safeValue >= 0
                        ? `rgba(32, 190, 169, ${0.12 + intensity * 0.72})`
                        : `rgba(93, 133, 255, ${0.12 + intensity * 0.72})`
                      return (
                        <div
                          className="mi-heat-cell"
                          key={`${row.feature}-${index}`}
                          style={{ background }}
                          title={`${row.feature} × ${dataset.correlation_matrix[index].feature}: ${value ?? 'not available'}`}
                        >
                          {value == null ? '—' : value.toFixed(2)}
                        </div>
                      )
                    })}
                  </div>
                ))}
              </div>
            </div>
            <p className="mi-panel-footnote">
              Zero measurements in Glucose, BloodPressure, SkinThickness, Insulin and BMI are excluded pairwise as missing. Correlation indicates association, not causation.
            </p>
          </article>
        </>
      )}

      <footer className="mi-disclaimer">
        DiaPredict is an academic machine-learning research project. Its outputs are not medical diagnoses and must not replace professional medical advice.
      </footer>
    </section>
  )
}

export default ModelInsights
