import { useEffect, useState } from 'react';
import { AlertTriangle, RefreshCw, ShieldCheck } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import USAMap from '../components/USAMap';
import PageHeader from '../components/PageHeader';
import { fetchFairness } from '../services/api';

const percent = value => `${(value * 100).toFixed(1)}%`;
const tooltipFormatter = value => [percent(value), 'Rate'];

export default function Fairness() {
  const [metrics, setMetrics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedState, setSelectedState] = useState(null);
  const [viewMode, setViewMode] = useState('map'); // 'map' or 'graph'

  useEffect(() => {
    let active = true;
    async function loadMetrics() {
      try {
        const data = await fetchFairness();
        const formattedData = Array.isArray(data)
          ? data.map(row => ({
            state_group: row.state_group,
            accuracy: parseFloat(row.accuracy),
            selection_rate: parseFloat(row.selection_rate),
            false_positive_rate: parseFloat(row.false_positive_rate),
            false_negative_rate: parseFloat(row.false_negative_rate),
            precision: parseFloat(row.precision),
            recall: parseFloat(row.recall)
          }))
          : [];
        if (active) setMetrics(formattedData);
      } catch (requestError) {
        if (active) setError(requestError.message);
      } finally {
        if (active) setLoading(false);
      }
    }
    loadMetrics();
    return () => { active = false; };
  }, []);

  const selectionMetrics = [...metrics].sort((left, right) => right.selection_rate - left.selection_rate);

  const validMetrics = metrics.filter(m => m.state_group && m.state_group !== 'Other');
  const selectionRates = validMetrics.map(m => m.selection_rate);
  const minSR = selectionRates.length ? Math.min(...selectionRates) : 0;
  const maxSR = selectionRates.length ? Math.max(...selectionRates) : 1;

  // Build customize object for react-usa-map
  const mapCustomize = {};
  validMetrics.forEach(row => {
    const range = maxSR - minSR || 1;
    const opacity = 0.15 + 0.85 * (row.selection_rate - minSR) / range;
    mapCustomize[row.state_group] = {
      fill: `rgba(252, 209, 22, ${opacity.toFixed(3)})`
    };
  });

  const handleMapClick = (event) => {
    const stateCode = event.target.dataset.name;
    const stateData = metrics.find(m => m.state_group === stateCode);
    if (stateData) {
      setSelectedState(stateData);
    }
  };

  const activeStateData = selectedState || metrics.find(m => m.state_group === 'CA') || metrics[0] || {
    state_group: 'N/A',
    selection_rate: 0,
    accuracy: 0,
    false_positive_rate: 0,
    false_negative_rate: 0,
    precision: 0,
    recall: 0
  };

  return <>
    <PageHeader eyebrow="Responsible AI" title="Fairness analysis" description="Compare loan decisions across applicant groups to check that outcomes stay consistent and fair." />
    {loading && <section className="panel fairness-status"><ShieldCheck size={22} /><span>Loading fairness metrics...</span></section>}
    {!loading && error && <section className="panel fairness-status fairness-error"><AlertTriangle size={22} /><div><strong>Unable to load metrics</strong><p>{error}</p><button type="button" onClick={() => window.location.reload()}><RefreshCw size={15} /> Retry</button></div></section>}
    {!loading && !error && !metrics.length && <section className="panel fairness-status"><ShieldCheck size={22} /><span>No fairness metrics are available.</span></section>}
    {!loading && !error && metrics.length > 0 && <div className="fairness-grid">
      <section className="panel chart-panel incorrect-decisions-panel" style={{ gridColumn: '1 / -1' }}>
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Incorrect analysis</p>
            <h2>Incorrect decisions by state</h2>
          </div>
          <span className="chart-note">Lower is better</span>
        </div>
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={metrics} margin={{ top: 12, right: 8, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#cbd5e1" />
              <XAxis
                dataKey="state_group"
                angle={-90}
                textAnchor="end"
                interval={0}
                height={120}
                tick={{ fontSize: 10, dy: 8, fill: '#475569' }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                tickFormatter={(tick) => `${Math.round(Number(tick) * 100)}%`}
                domain={[0, 'auto']}
                tickLine={false}
                axisLine={false}
                width={50}
                tick={{ fill: '#475569' }}
              />
              <Tooltip formatter={tooltipFormatter} contentStyle={{ borderRadius: 8, border: '1px solid #cbd5e1', backgroundColor: '#ffffff', color: '#000000' }} />
              <Bar dataKey="false_positive_rate" name="Wrongly flagged as risky" fill="#fcd116" radius={[4, 4, 0, 0]} />
              <Bar dataKey="false_negative_rate" name="Wrongly cleared as low-risk" fill="#0f0f11" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', marginTop: '16px', fontSize: '12px', color: '#475569', fontWeight: 600 }}>
          <span style={{ display: 'inline-block', width: '10px', height: '10px', backgroundColor: '#fcd116', borderRadius: '2px' }}></span>
          <span>Metric A: Yellow (A-state data), Black (B-state data)</span>
        </div>
      </section>

      <section className="panel chart-panel" style={{ gridColumn: '1 / -1' }}>
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Selection rate</p>
            <h2>Approval selection by state</h2>
          </div>
          <div className="view-toggle-buttons">
            <button
              type="button"
              className={`view-toggle-btn ${viewMode === 'map' ? 'active' : ''}`}
              onClick={() => setViewMode('map')}
            >
              Map View
            </button>
            <button
              type="button"
              className={`view-toggle-btn ${viewMode === 'graph' ? 'active' : ''}`}
              onClick={() => setViewMode('graph')}
            >
              Graph View
            </button>
          </div>
        </div>

        {viewMode === 'map' ? (
          <div className="map-layout">
            <div>
              <div className="map-container">
                <USAMap customize={mapCustomize} onClick={handleMapClick} />
              </div>
              <div className="map-legend">
                <span>Lower selection rate</span>
                <div className="map-legend-gradient"></div>
                <span>Higher selection rate</span>
              </div>
            </div>
            <div className="map-details-card">
              <h3>State Details: {activeStateData.state_group}</h3>
              <div className="map-details-list">
                <div className="map-details-item">
                  <span>Selection rate:</span>
                  <strong>{(activeStateData.selection_rate * 100).toFixed(1)}%</strong>
                </div>
                <div className="map-details-item">
                  <span>Accuracy:</span>
                  <strong>{(activeStateData.accuracy * 100).toFixed(1)}%</strong>
                </div>
                <div className="map-details-item">
                  <span>Wrongly flagged as risky (FPR):</span>
                  <strong>{(activeStateData.false_positive_rate * 100).toFixed(1)}%</strong>
                </div>
                <div className="map-details-item">
                  <span>Wrongly cleared as low-risk (FNR):</span>
                  <strong>{(activeStateData.false_negative_rate * 100).toFixed(1)}%</strong>
                </div>
                <div className="map-details-item">
                  <span>Model precision:</span>
                  <strong>{(activeStateData.precision * 100).toFixed(1)}%</strong>
                </div>
                <div className="map-details-item">
                  <span>Model recall:</span>
                  <strong>{(activeStateData.recall * 100).toFixed(1)}%</strong>
                </div>
              </div>
              <p className="muted" style={{ marginTop: '16px', fontSize: '11px', lineHeight: '1.4' }}>
                Click on any state on the map to display its detailed decision and fairness metrics in this panel.
              </p>
            </div>
          </div>
        ) : (
          <div className="chart-wrap chart-wrap-vertical">
            <ResponsiveContainer width="100%" height={860}>
              <BarChart data={selectionMetrics} layout="vertical" margin={{ top: 4, right: 16, left: 4, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#cbd5e1" />
                <XAxis type="number" domain={[0, 'auto']} tickFormatter={(tick) => `${(tick * 100).toFixed(0)}%`} axisLine={false} tickLine={false} tick={{ fill: '#475569' }} />
                <YAxis type="category" dataKey="state_group" axisLine={false} tickLine={false} width={38} interval={0} tick={{ fill: '#475569' }} />
                <Tooltip formatter={tooltipFormatter} contentStyle={{ borderRadius: 8, border: '1px solid #cbd5e1', backgroundColor: '#ffffff', color: '#000000' }} />
                <Bar dataKey="selection_rate" name="Selection rate" fill="#fcd116" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </section>
    </div>}
  </>;
}
