import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const API_BASE = 'http://127.0.0.1:8000/api'
function toClassCountRows(points) {
  const byActual = new Map()
  const byPred = new Map()

  points.forEach((row) => {
    byActual.set(row.actual, (byActual.get(row.actual) || 0) + 1)
    byPred.set(row.predicted, (byPred.get(row.predicted) || 0) + 1)
  })

  const labels = Array.from(new Set([...byActual.keys(), ...byPred.keys()]))
  return labels.map((label) => ({
    label,
    actual: byActual.get(label) || 0,
    predicted: byPred.get(label) || 0,
  }))
}

function toErrorBins(errors, binCount = 18) {
  if (!errors || errors.length === 0) return []

  const min = Math.min(...errors)
  const max = Math.max(...errors)
  if (min === max) {
    return [{ range: `${min.toFixed(2)}`, count: errors.length }]
  }

  const step = (max - min) / binCount
  const bins = Array.from({ length: binCount }, (_, i) => ({
    start: min + i * step,
    end: min + (i + 1) * step,
    count: 0,
  }))

  errors.forEach((v) => {
    const idx = Math.min(Math.floor((v - min) / step), binCount - 1)
    bins[idx].count += 1
  })

  return bins.map((b) => ({
    range: `${b.start.toFixed(2)} to ${b.end.toFixed(2)}`,
    count: b.count,
  }))
}

function toClassMetrics(points) {
  if (!points || points.length === 0) return []

  const labels = Array.from(new Set(points.flatMap((p) => [p.actual, p.predicted])))
  return labels.map((label) => {
    let tp = 0
    let fp = 0
    let fn = 0
    let support = 0

    points.forEach((p) => {
      const isActual = p.actual === label
      const isPred = p.predicted === label
      if (isActual) support += 1
      if (isActual && isPred) tp += 1
      else if (!isActual && isPred) fp += 1
      else if (isActual && !isPred) fn += 1
    })

    const precision = tp + fp === 0 ? 0 : tp / (tp + fp)
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn)
    const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall)

    return { label, precision, recall, f1, support }
  })
}

function HeatCell({ entry }) {
  const intensity = Math.round(Math.abs(entry.value) * 255)
  const bg = entry.value >= 0 ? `rgb(${230 - intensity / 4}, 255, ${230 - intensity / 3})` : `rgb(255, ${230 - intensity / 3}, ${230 - intensity / 4})`
  return (
    <div className="heat-cell" style={{ background: bg }}>
      <div>{entry.x}</div>
      <div>{entry.y}</div>
      <strong>{Number(entry.value).toFixed(2)}</strong>
    </div>
  )
}

export default function App() {
  const [threshold, setThreshold] = useState(50)
  const [showAdvancedLoad, setShowAdvancedLoad] = useState(false)
  const [uploadedDatasets, setUploadedDatasets] = useState([])
  const [selectedDatasetId, setSelectedDatasetId] = useState('')
  const [isUploadDragOver, setIsUploadDragOver] = useState(false)
  const [themePreference, setThemePreference] = useState(() => {
    const saved = localStorage.getItem('themePreference')
    return saved === 'classic' || saved === 'green' ? saved : 'classic'
  })
  const [resolvedTheme, setResolvedTheme] = useState('classic')
  const uploadInputRef = useRef(null)

  const [loadInfo, setLoadInfo] = useState(null)
  const [trainMode, setTrainMode] = useState('both')
  const [autoSelectBestModels, setAutoSelectBestModels] = useState(false)
  const [classTarget, setClassTarget] = useState('')
  const [regTarget, setRegTarget] = useState('')
  const [classModel, setClassModel] = useState('random_forest')
  const [regModel, setRegModel] = useState('random_forest')
  const [trainInfo, setTrainInfo] = useState(null)

  const [classInput, setClassInput] = useState({})
  const [regInput, setRegInput] = useState({})

  const [classPrediction, setClassPrediction] = useState('')
  const [regPrediction, setRegPrediction] = useState('')
  const [previewSearch, setPreviewSearch] = useState('')
  const [previewFilters, setPreviewFilters] = useState({})

  const [loading, setLoading] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [error, setError] = useState('')
  const reportRef = useRef(null)
  const trainedReportRef = useRef(null)
  const chartColors = resolvedTheme === 'green'
    ? { grid: '#bcd7cd', axis: '#3f6a5a', text: '#214437' }
    : { grid: '#c8d8ec', axis: '#5f7ea4', text: '#2f4e72' }

  const classCountRows = useMemo(() => {
    if (!trainInfo?.classification?.points) return []
    return toClassCountRows(trainInfo.classification.points)
  }, [trainInfo])

  const errorBins = useMemo(() => {
    if (!trainInfo?.regression?.errors) return []
    return toErrorBins(trainInfo.regression.errors)
  }, [trainInfo])

  const classMetricRows = useMemo(() => {
    const rows = trainInfo?.classification?.class_metrics
    if (rows && rows.length > 0) return rows
    return toClassMetrics(trainInfo?.classification?.points || [])
  }, [trainInfo])

  const classDistributionRows = useMemo(() => {
    const rows = trainInfo?.summary?.class_distribution || []
    return rows.filter((row) => {
      const label = String(row.label || '').trim().toLowerCase()
      return label !== 'unknown'
    })
  }, [trainInfo])

  const executiveNarrative = useMemo(() => {
    if (!trainInfo) return null

    const distribution = [...classDistributionRows]
      .sort((a, b) => Number(b.count || 0) - Number(a.count || 0))
      .slice(0, 3)

    const distText = distribution.length > 0
      ? distribution.map((row) => `${row.label}`).join(', ')
      : 'no strong class concentration detected'

    const corr = trainInfo?.summary?.correlation || []
    const churnPairs = corr.filter((row) => {
      const x = String(row.x || '').toLowerCase()
      const y = String(row.y || '').toLowerCase()
      return x !== y && (x.includes('churn') || y.includes('churn'))
    })
    const strongestNegative = [...churnPairs]
      .filter((row) => Number(row.value) < 0)
      .sort((a, b) => Number(a.value) - Number(b.value))[0]
    const strongestPositive = [...churnPairs]
      .filter((row) => Number(row.value) > 0)
      .sort((a, b) => Number(b.value) - Number(a.value))[0]

    const strongestNegativeText = strongestNegative
      ? `${strongestNegative.x} vs ${strongestNegative.y} (${Number(strongestNegative.value).toFixed(2)})`
      : 'no strong negative churn correlation'

    const strengths = [
      `Large concentration in ${distText} creates scale for conversion and upsell programs, improving premium revenue growth.`,
      'Clear churn/engagement drivers enable targeted offers that protect retention and lift repeat purchases.',
      'Segment-level targeting supports more efficient campaign spend and better ROI.',
    ]

    const weaknesses = [
      'Customer mix remains concentrated in lower tiers, which limits premium revenue share.',
      'Class-level prediction imbalance can cause uneven campaign targeting and wasted spend.',
      `Churn sensitivity is concentrated around ${strongestNegativeText}, meaning other drivers may be under-captured.`,
    ]

    const suggestions = [
      'Build weekly churn watchlists and run proactive save campaigns for high-risk cohorts.',
      'Prioritize low wallet-point and low transaction-value customers with targeted booster offers.',
      'Run A/B tests by segment on offer type, channel, and timing to improve conversion yield.',
    ]

    const development = [
      'Design tier-up journeys from entry segments to premium segments using staged rewards.',
      'Add richer behavioral features (recency, frequency, response history) to improve model lift.',
      'Set drift monitoring and retraining cadence to keep targeting quality stable over time.',
    ]

    const growthActions = [
      'Higher premium revenue from focused tier-up of No Membership and Basic Membership.',
      'Lower churn by targeting low points_in_wallet customers with retention offers.',
      'Higher monthly revenue per user by lifting low avg_transaction_value segments.',
      'Better recurring revenue protection through weekly high-risk customer save lists.',
      'Higher campaign ROI via monthly KPI-led budget reallocation to best-performing segments.',
    ]

    const totalCustomers = loadInfo?.rows ?? null
    const totalClassCount = classDistributionRows.reduce((sum, row) => sum + Number(row.count || 0), 0)
    const topSegment = distribution[0]?.label ? String(distribution[0].label) : 'n/a'
    const topShare = totalClassCount > 0
      ? `${((Number(distribution[0]?.count || 0) / totalClassCount) * 100).toFixed(1)}%`
      : 'n/a'
    const top2Share = totalClassCount > 0
      ? `${((Number(distribution[0]?.count || 0) + Number(distribution[1]?.count || 0)) / totalClassCount * 100).toFixed(1)}%`
      : 'n/a'
    const premiumKeywords = ['gold', 'platinum', 'premium', 'silver']
    const baseKeywords = ['no membership', 'basic', 'entry', 'free']
    const premiumCount = classDistributionRows
      .filter((row) => premiumKeywords.some((k) => String(row.label || '').toLowerCase().includes(k)))
      .reduce((sum, row) => sum + Number(row.count || 0), 0)
    const baseCount = classDistributionRows
      .filter((row) => baseKeywords.some((k) => String(row.label || '').toLowerCase().includes(k)))
      .reduce((sum, row) => sum + Number(row.count || 0), 0)
    const premiumShare = totalClassCount > 0 ? `${((premiumCount / totalClassCount) * 100).toFixed(1)}%` : 'n/a'
    const baseShare = totalClassCount > 0 ? `${((baseCount / totalClassCount) * 100).toFixed(1)}%` : 'n/a'
    const kpis = [
      `Total customers: ${totalCustomers != null ? Number(totalCustomers).toLocaleString() : 'n/a'}`,
      `Top segment share: ${topSegment} (${topShare})`,
      `Top 2 segments share: ${top2Share}`,
      `Premium-tier share: ${premiumShare}`,
      `Base-tier share: ${baseShare}`,
    ]
    const modelKpis = [
      `Targeting accuracy: ${trainInfo?.classification?.accuracy != null ? `${(Number(trainInfo.classification.accuracy) * 100).toFixed(2)}%` : 'n/a'}`,
      `Mean F1 score: ${trainInfo?.classification?.mean_f1 != null ? Number(trainInfo.classification.mean_f1).toFixed(3) : 'n/a'}`,
      `Forecast R2: ${trainInfo?.regression?.r2 != null ? Number(trainInfo.regression.r2).toFixed(3) : 'n/a'}`,
      `Forecast MAE: ${trainInfo?.regression?.mae != null ? Number(trainInfo.regression.mae).toFixed(3) : 'n/a'}`,
    ]

    return {
      narrative:
        `Current business picture shows concentration in key membership bands (${distText}) with actionable churn signals for targeted growth. ` +
        `Top risk-related relationship is ${strongestNegativeText}, which supports focused retention actions while scaling tier-up conversion programs.`,
      strengths,
      weaknesses,
      suggestions,
      development,
      growthActions,
      kpis,
      modelKpis,
    }
  }, [classDistributionRows, loadInfo, trainInfo])

  const classColor = (label) => {
    const v = String(label || '').trim().toLowerCase()
    if (v === 'f' || v === 'female') return '#f9a8d4'
    if (v === 'm' || v === 'male') return '#93c5fd'
    return '#0f766e'
  }

  const canTrainClassification = (loadInfo?.classification_targets?.length || 0) > 0
  const canTrainRegression = (loadInfo?.regression_targets?.length || 0) > 0
  const showClassification = Boolean(trainInfo?.classification)
  const showRegression = Boolean(trainInfo?.regression)
  const classModelOptions = loadInfo?.classification_models || [
    'random_forest',
    'extra_trees',
    'decision_tree',
    'gradient_boosting',
    'logistic_regression',
    'linear_svc',
    'k_neighbors',
  ]
  const regModelOptions = loadInfo?.regression_models || [
    'random_forest',
    'extra_trees',
    'decision_tree',
    'gradient_boosting',
    'linear_regression',
    'ridge',
    'lasso',
    'elastic_net',
    'linear_svr',
    'k_neighbors',
  ]

  const formatModelName = (name) => String(name || '').split('_').map((p) => p.charAt(0).toUpperCase() + p.slice(1)).join(' ')
  const previewColumns = useMemo(() => Object.keys(loadInfo?.preview?.[0] || {}), [loadInfo])

  const previewFilterOptions = useMemo(() => {
    const rows = loadInfo?.preview || []
    const cols = Object.keys(rows[0] || {})
    const options = {}
    cols.forEach((col) => {
      options[col] = Array.from(new Set(rows.map((r) => String(r[col] ?? '').trim()).filter(Boolean))).slice(0, 100)
    })
    return options
  }, [loadInfo])

  const previewFilterColumns = useMemo(
    () => previewColumns.filter((col) => (previewFilterOptions?.[col]?.length || 0) > 0).slice(0, 5),
    [previewColumns, previewFilterOptions]
  )

  const filteredPreviewRows = useMemo(() => {
    const rows = loadInfo?.preview || []
    const q = previewSearch.trim().toLowerCase()
    return rows.filter((row) => {
      if (q) {
        const hasText = Object.values(row).some((v) => String(v ?? '').toLowerCase().includes(q))
        if (!hasText) return false
      }
      for (const [col, selected] of Object.entries(previewFilters)) {
        if (!selected || selected === 'All') continue
        if (String(row[col] ?? '') !== selected) return false
      }
      return true
    })
  }, [loadInfo, previewFilters, previewSearch])

  const resetPreviewFilters = () => {
    setPreviewSearch('')
    setPreviewFilters({})
  }

  const applyLoadedDatasetInfo = (data) => {
    setLoadInfo(data)
    setClassTarget(data.classification_targets?.[0] || '')
    setRegTarget(data.regression_targets?.[0] || '')
    setClassModel(data.classification_models?.[0] || 'random_forest')
    setRegModel(data.regression_models?.[0] || 'random_forest')
    setAutoSelectBestModels(false)
    if (data.classification_targets?.length && data.regression_targets?.length) setTrainMode('both')
    else if (data.classification_targets?.length) setTrainMode('classification')
    else if (data.regression_targets?.length) setTrainMode('regression')
  }

  const exportFilteredPreviewCsv = () => {
    if (!previewColumns.length || !filteredPreviewRows.length) return

    const escapeCell = (value) => `"${String(value ?? '').replaceAll('"', '""')}"`
    const lines = [
      previewColumns.join(','),
      ...filteredPreviewRows.map((row) => previewColumns.map((col) => escapeCell(row[col])).join(',')),
    ]
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'filtered_preview.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  const exportFilteredPreviewExcel = () => {
    if (!previewColumns.length || !filteredPreviewRows.length) return

    const escapeHtml = (value) =>
      String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')

    const headerHtml = `<tr>${previewColumns.map((col) => `<th>${escapeHtml(col)}</th>`).join('')}</tr>`
    const bodyHtml = filteredPreviewRows
      .map((row) => `<tr>${previewColumns.map((col) => `<td>${escapeHtml(row[col])}</td>`).join('')}</tr>`)
      .join('')

    const html = `
      <html>
      <head><meta charset="utf-8" /></head>
      <body>
        <table border="1">
          <thead>${headerHtml}</thead>
          <tbody>${bodyHtml}</tbody>
        </table>
      </body>
      </html>
    `

    const blob = new Blob([html], { type: 'application/vnd.ms-excel;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'filtered_preview.xls'
    a.click()
    URL.revokeObjectURL(url)
  }

  useEffect(() => {
    const schema = trainInfo?.classification?.form_schema || []
    const initial = {}
    schema.forEach((field) => {
      initial[field.name] = field.default
    })
    setClassInput(initial)
  }, [trainInfo?.classification?.form_schema])

  useEffect(() => {
    const schema = trainInfo?.regression?.form_schema || []
    const initial = {}
    schema.forEach((field) => {
      initial[field.name] = field.default
    })
    setRegInput(initial)
  }, [trainInfo?.regression?.form_schema])

  useEffect(() => {
    const next = themePreference || 'classic'
    document.documentElement.setAttribute('data-theme', next)
    setResolvedTheme(next)
    localStorage.setItem('themePreference', next)
  }, [themePreference])

  useEffect(() => {
    resetPreviewFilters()
  }, [loadInfo])

  async function loadDataset() {
    setLoading(true)
    setError('')
    setTrainInfo(null)
    setClassPrediction('')
    setRegPrediction('')

    try {
      const selectedUpload = uploadedDatasets.find((item) => item.id === selectedDatasetId)
      if (!selectedUpload?.file) {
        throw new Error('Please upload and select a dataset first.')
      }
      const formData = new FormData()
      formData.append('file', selectedUpload.file)
      formData.append('high_cardinality_threshold', String(Number(threshold)))
      const res = await fetch(`${API_BASE}/load/upload`, {
        method: 'POST',
        body: formData,
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Failed to load dataset')
      if (selectedUpload?.file) {
        setUploadedDatasets((prev) =>
          prev.map((item) => (item.id === selectedUpload.id ? { ...item, rows: data.rows } : item))
        )
      }
      applyLoadedDatasetInfo(data)
    } catch (e) {
      const msg = e?.message === 'Failed to fetch'
        ? 'Cannot connect to backend API. Start run_backend.bat first, then refresh.'
        : e.message
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  function addUploadedFiles(files) {
    if (!files.length) return

    setError('')
    let lastId = ''
    setUploadedDatasets((prev) => {
      const next = [...prev]
      files.forEach((file) => {
        const id = `${file.name}-${file.size}-${file.lastModified}`
        lastId = id
        const idx = next.findIndex((item) => item.id === id)
        const row = { id, name: file.name, rows: null, file }
        if (idx >= 0) next[idx] = row
        else next.push(row)
      })
      return next
    })
    if (lastId) setSelectedDatasetId(lastId)
  }

  async function uploadDatasets(event) {
    const files = Array.from(event.target.files || [])
    addUploadedFiles(files)
    if (event.target) event.target.value = ''
  }

  function handleUploadDragOver(event) {
    event.preventDefault()
    setIsUploadDragOver(true)
  }

  function handleUploadDragLeave(event) {
    event.preventDefault()
    setIsUploadDragOver(false)
  }

  function handleUploadDrop(event) {
    event.preventDefault()
    setIsUploadDragOver(false)
    const files = Array.from(event.dataTransfer?.files || []).filter((file) =>
      /\.(csv|xlsx|xls)$/i.test(file.name || '')
    )
    if (!files.length) {
      setError('Only CSV/XLS/XLSX files are supported.')
      return
    }
    addUploadedFiles(files)
  }

  function removeDataset(id) {
    setUploadedDatasets((prev) => {
      const next = prev.filter((item) => item.id !== id)
      if (selectedDatasetId === id) setSelectedDatasetId(next[0]?.id || '')
      return next
    })
  }

  function clearDatasets() {
    setUploadedDatasets([])
    setSelectedDatasetId('')
  }

  const exportFileStamp = () => new Date().toISOString().replaceAll(':', '-').slice(0, 19)

  async function exportDashboardAsImage() {
    if (!trainedReportRef.current) {
      setError('Train the model first, then export the analytics report.')
      return
    }
    setExporting(true)
    try {
      const html2canvas = (await import('html2canvas')).default
      const canvas = await html2canvas(trainedReportRef.current, {
        scale: 2,
        useCORS: true,
        backgroundColor: resolvedTheme === 'green' ? '#eef5f2' : '#f2f5fb',
        windowWidth: trainedReportRef.current.scrollWidth,
        windowHeight: trainedReportRef.current.scrollHeight,
      })
      const url = canvas.toDataURL('image/png')
      const link = document.createElement('a')
      link.href = url
      link.download = `dashboard-${exportFileStamp()}.png`
      link.click()
    } catch (e) {
      setError(e?.message || 'Failed to export image')
    } finally {
      setExporting(false)
    }
  }

  async function exportDashboardAsPdf() {
    if (!trainedReportRef.current) {
      setError('Train the model first, then export the analytics report.')
      return
    }
    setExporting(true)
    try {
      const html2canvas = (await import('html2canvas')).default
      const { jsPDF } = await import('jspdf')
      const canvas = await html2canvas(trainedReportRef.current, {
        scale: 2,
        useCORS: true,
        backgroundColor: resolvedTheme === 'green' ? '#eef5f2' : '#f2f5fb',
        windowWidth: trainedReportRef.current.scrollWidth,
        windowHeight: trainedReportRef.current.scrollHeight,
      })
      const imageData = canvas.toDataURL('image/png')
      const pdf = new jsPDF('p', 'mm', 'a4')
      const pageWidth = pdf.internal.pageSize.getWidth()
      const pageHeight = pdf.internal.pageSize.getHeight()
      const imageWidth = pageWidth
      const imageHeight = (canvas.height * imageWidth) / canvas.width

      let heightLeft = imageHeight
      let position = 0
      pdf.addImage(imageData, 'PNG', 0, position, imageWidth, imageHeight)
      heightLeft -= pageHeight

      while (heightLeft > 0) {
        position = heightLeft - imageHeight
        pdf.addPage()
        pdf.addImage(imageData, 'PNG', 0, position, imageWidth, imageHeight)
        heightLeft -= pageHeight
      }

      pdf.save(`dashboard-${exportFileStamp()}.pdf`)
    } catch (e) {
      setError(e?.message || 'Failed to export PDF')
    } finally {
      setExporting(false)
    }
  }

  async function trainModels() {
    if (trainMode === 'both' && (!classTarget || !regTarget)) {
      setError('Please select both classification and regression targets.')
      return
    }
    if (trainMode === 'classification' && !classTarget) {
      setError('Please select a classification target.')
      return
    }
    if (trainMode === 'regression' && !regTarget) {
      setError('Please select a regression target.')
      return
    }

    setLoading(true)
    setError('')
    setClassPrediction('')
    setRegPrediction('')

    try {
      const res = await fetch(`${API_BASE}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: trainMode,
          classification_target: trainMode !== 'regression' ? classTarget : null,
          regression_target: trainMode !== 'classification' ? regTarget : null,
          classification_model: trainMode !== 'regression' ? classModel : 'random_forest',
          regression_model: trainMode !== 'classification' ? regModel : 'random_forest',
          auto_select_best_models: autoSelectBestModels,
        }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Failed to train models')
      setTrainInfo(data)
    } catch (e) {
      const msg = e?.message === 'Failed to fetch'
        ? 'Cannot connect to backend API. Start run_backend.bat first, then refresh.'
        : e.message
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  async function predictClassification() {
    setError('')
    try {
      const res = await fetch(`${API_BASE}/predict/classification`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ values: classInput }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Failed classification prediction')
      setClassPrediction(`${data.target}: ${data.prediction}`)
    } catch (e) {
      const msg = e?.message === 'Failed to fetch'
        ? 'Cannot connect to backend API. Start run_backend.bat first, then refresh.'
        : e.message
      setError(msg)
    }
  }

  async function predictRegression() {
    setError('')
    try {
      const res = await fetch(`${API_BASE}/predict/regression`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ values: regInput }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Failed regression prediction')
      setRegPrediction(`${data.target}: ${Number(data.prediction).toFixed(4)}`)
    } catch (e) {
      const msg = e?.message === 'Failed to fetch'
        ? 'Cannot connect to backend API. Start run_backend.bat first, then refresh.'
        : e.message
      setError(msg)
    }
  }

  return (
    <div className="app-shell">
      <div className="bg-orb orb-a" />
      <div className="bg-orb orb-b" />
      <div className="bg-grid" />

      <div className="container" ref={reportRef}>
        <header className="hero">
          <div className="hero-title">
            <h1>NOVASIGHT</h1>
            <p className="eyebrow">The Intelligence Studio</p>
            <p className="hero-copy">
              Load your dataset, train classification and regression models, and inspect quality metrics in one
              professional workspace.
            </p>
          </div>
          <div className="hero-tags">
            <span>FastAPI Backend</span>
            <span>React + Recharts</span>
            <span>Live Predictions</span>
            <div className="theme-switch">
              <label htmlFor="theme-select">Theme</label>
              <select
                id="theme-select"
                value={themePreference}
                onChange={(e) => setThemePreference(e.target.value)}
              >
                <option value="classic">Executive Ivory</option>
                <option value="green">Green Finance</option>
              </select>
            </div>
          </div>
        </header>

        {error && <div className="status error">{error}</div>}
        {loading && (
          <div className="status status-loading">
            <span className="spiral-loader" aria-hidden="true" />
            <span>Processing request...</span>
          </div>
        )}
        {exporting && (
          <div className="status status-loading">
            <span className="spiral-loader" aria-hidden="true" />
            <span>Preparing export...</span>
          </div>
        )}

        {trainInfo && (
          <div className="report-actions">
            <button className="btn-secondary" type="button" onClick={exportDashboardAsImage} disabled={exporting || loading}>
              Save as Image
            </button>
            <button className="btn-primary" type="button" onClick={exportDashboardAsPdf} disabled={exporting || loading}>
              Save as PDF
            </button>
          </div>
        )}

        <section className="panel controls">
          <div className="dataset-manager">
            <div className="dataset-manager-top">
              <div>
                <h4>1. Dataset Manager</h4>
                <p className="dataset-manager-help">Upload multiple files, click a row to select, then load.</p>
              </div>
              <div className="dataset-manager-controls">
                <select
                  value={selectedDatasetId}
                  onChange={(e) => setSelectedDatasetId(e.target.value)}
                >
                  <option value="">Select uploaded dataset</option>
                  {uploadedDatasets.map((item) => (
                    <option key={item.id} value={item.id}>{item.name}</option>
                  ))}
                </select>
                <button
                  className="btn-primary"
                  type="button"
                  onClick={() => uploadInputRef.current?.click()}
                  disabled={loading}
                >
                  Browse Files
                </button>
                {uploadedDatasets.length > 0 && (
                  <button className="btn-secondary" type="button" onClick={clearDatasets} disabled={loading}>
                    Clear All
                  </button>
                )}
                <input
                  ref={uploadInputRef}
                  className="hidden-upload"
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  multiple
                  onChange={uploadDatasets}
                />
              </div>
            </div>
            <div
              className={`upload-dropzone ${isUploadDragOver ? 'drag-over' : ''}`}
              onDragOver={handleUploadDragOver}
              onDragLeave={handleUploadDragLeave}
              onDrop={handleUploadDrop}
              onClick={() => uploadInputRef.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  uploadInputRef.current?.click()
                }
              }}
            >
              <strong>Drop CSV/XLS/XLSX files here</strong>
              <p>or click to browse from your computer</p>
            </div>
            <p className="dataset-manager-count">Added {uploadedDatasets.length} file(s)</p>
            <div className="dataset-list">
              {uploadedDatasets.length === 0 ? (
                <div className="dataset-empty">No files uploaded yet.</div>
              ) : (
                uploadedDatasets.map((item) => (
                  <div
                    key={item.id}
                    className={`dataset-row ${item.id === selectedDatasetId ? 'active' : ''}`}
                    role="button"
                    tabIndex={0}
                    onClick={() => setSelectedDatasetId(item.id)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        setSelectedDatasetId(item.id)
                      }
                    }}
                  >
                    <div>
                      <strong>{item.name}</strong>
                      <p>{item.rows == null ? 'Ready to load' : `${Number(item.rows).toLocaleString()} rows loaded`}</p>
                    </div>
                    <button
                      className="btn-secondary"
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        removeDataset(item.id)
                      }}
                    >
                      Remove
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>

          <button
            className="btn-secondary"
            type="button"
            onClick={() => setShowAdvancedLoad((prev) => !prev)}
          >
            {showAdvancedLoad ? 'Hide Advanced Settings' : 'Show Advanced Settings'}
          </button>

          {showAdvancedLoad && (
            <div className="form-grid">
              <div className="input-row">
                <label>High-cardinality threshold</label>
                <input type="number" value={threshold} onChange={(e) => setThreshold(e.target.value)} />
              </div>
            </div>
          )}

          <button className="btn-primary" onClick={loadDataset} disabled={loading}>
            Load Dataset
          </button>
        </section>

        {loadInfo && (
          <>
            <section className="kpi-grid">
              <article className="kpi-card">
                <p>Total Rows</p>
                <h3>{loadInfo.rows}</h3>
              </article>
              <article className="kpi-card">
                <p>Total Columns</p>
                <h3>{loadInfo.columns}</h3>
              </article>
              <article className="kpi-card">
                <p>Dropped Columns</p>
                <h3>{loadInfo.dropped_columns.length}</h3>
              </article>
            </section>

            <section className="panel">
              <div className="panel-head">
                <h3>2. Model Targets</h3>
                <p>Pick a training mode based on dataset type and train only supported tasks.</p>
              </div>

              <div className="form-grid">
                <div className="input-row">
                  <label>Training mode</label>
                  <select value={trainMode} onChange={(e) => setTrainMode(e.target.value)}>
                    <option value="both" disabled={!(canTrainClassification && canTrainRegression)}>Both</option>
                    <option value="classification" disabled={!canTrainClassification}>Classification only</option>
                    <option value="regression" disabled={!canTrainRegression}>Regression only</option>
                  </select>
                </div>
                <div className="input-row">
                  <label>Model selection strategy</label>
                  <select value={autoSelectBestModels ? 'auto' : 'manual'} onChange={(e) => setAutoSelectBestModels(e.target.value === 'auto')}>
                    <option value="manual">Manual (use selected models)</option>
                    <option value="auto">Auto (train all and pick best)</option>
                  </select>
                </div>
                <div className="input-row">
                  <label>Classification target</label>
                  <select
                    value={classTarget}
                    onChange={(e) => setClassTarget(e.target.value)}
                    disabled={!canTrainClassification || trainMode === 'regression'}
                  >
                    {loadInfo.classification_targets.map((col) => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>
                <div className="input-row">
                  <label>Classification model</label>
                  <select
                    value={autoSelectBestModels && trainInfo?.classification?.model_name ? trainInfo.classification.model_name : classModel}
                    onChange={(e) => setClassModel(e.target.value)}
                    disabled={!canTrainClassification || trainMode === 'regression' || autoSelectBestModels}
                  >
                    {classModelOptions.map((modelName) => (
                      <option key={modelName} value={modelName}>{formatModelName(modelName)}</option>
                    ))}
                  </select>
                </div>
                <div className="input-row">
                  <label>Regression target</label>
                  <select
                    value={regTarget}
                    onChange={(e) => setRegTarget(e.target.value)}
                    disabled={!canTrainRegression || trainMode === 'classification'}
                  >
                    {loadInfo.regression_targets.map((col) => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>
                <div className="input-row">
                  <label>Regression model</label>
                  <select
                    value={autoSelectBestModels && trainInfo?.regression?.model_name ? trainInfo.regression.model_name : regModel}
                    onChange={(e) => setRegModel(e.target.value)}
                    disabled={!canTrainRegression || trainMode === 'classification' || autoSelectBestModels}
                  >
                    {regModelOptions.map((modelName) => (
                      <option key={modelName} value={modelName}>{formatModelName(modelName)}</option>
                    ))}
                  </select>
                </div>
              </div>

              <button className="btn-primary" onClick={trainModels} disabled={loading}>
                Train Models
              </button>
            </section>

            <section className="panel">
              <div className="panel-head">
                <h3>Data Preview</h3>
                <p>First 10 rows after cleaning.</p>
              </div>

              <div className="preview-toolbar">
                <div className="input-row">
                  <label>Global Search</label>
                  <input
                    value={previewSearch}
                    onChange={(e) => setPreviewSearch(e.target.value)}
                    placeholder="Search across all columns"
                  />
                </div>

                {previewFilterColumns.map((col) => (
                  <div className="input-row" key={`preview-filter-${col}`}>
                    <label>{col.replaceAll('_', ' ')}</label>
                    <select
                      value={previewFilters[col] || 'All'}
                      onChange={(e) => setPreviewFilters((prev) => ({ ...prev, [col]: e.target.value }))}
                    >
                      <option value="All">All</option>
                      {(previewFilterOptions[col] || []).map((value) => (
                        <option key={`${col}-${value}`} value={value}>{value}</option>
                      ))}
                    </select>
                  </div>
                ))}

                <div className="preview-actions">
                  <button className="btn-secondary" onClick={resetPreviewFilters} type="button">Reset</button>
                  <button className="btn-primary" onClick={exportFilteredPreviewCsv} type="button">Export CSV</button>
                  <button className="btn-primary" onClick={exportFilteredPreviewExcel} type="button">Export Excel</button>
                </div>
              </div>

              <div className="table-wrap">
                <table className="preview-table">
                  <thead>
                    <tr>
                      {previewColumns.map((col) => (
                        <th key={col}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {filteredPreviewRows.map((row, idx) => (
                      <tr key={idx}>
                        {previewColumns.map((col) => (
                          <td key={col}>{String(row[col] ?? '')}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}

        {trainInfo && (
          <div ref={trainedReportRef}>
            {showClassification && (
              <>
                <h2 className="section-title">Classification Analytics</h2>
                <section className="kpi-grid">
                  <article className="kpi-card">
                    <p>Classification Accuracy</p>
                    <h3>{(trainInfo.classification.accuracy * 100).toFixed(2)}%</h3>
                  </article>
                  <article className="kpi-card">
                    <p>Model</p>
                    <h3>{formatModelName(trainInfo.classification.model_name)}</h3>
                  </article>
                </section>
                {trainInfo.classification.model_comparison?.length > 1 && (
                  <section className="panel">
                    <div className="panel-head">
                      <h3>Classification Model Comparison</h3>
                      <p>Sorted by accuracy (then mean F1).</p>
                    </div>
                    <div className="table-wrap">
                      <table className="preview-table">
                        <thead>
                          <tr>
                            <th>Model</th>
                            <th>Accuracy</th>
                            <th>Mean F1</th>
                          </tr>
                        </thead>
                        <tbody>
                          {trainInfo.classification.model_comparison.map((row) => (
                            <tr key={row.model_name}>
                              <td>{formatModelName(row.model_name)}</td>
                              <td>{(Number(row.accuracy) * 100).toFixed(2)}%</td>
                              <td>{Number(row.mean_f1).toFixed(4)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </section>
                )}

                <section className="grid charts-grid">
                  <article className="panel">
                    <div className="panel-head">
                      <h3>Actual vs Predicted Counts</h3>
                    </div>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={classCountRows}>
                        <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                        <XAxis dataKey="label" stroke={chartColors.axis} tick={{ fill: chartColors.text, fontSize: 12 }} />
                        <YAxis stroke={chartColors.axis} tick={{ fill: chartColors.text, fontSize: 12 }} />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="actual" fill="#00d1ff" />
                        <Bar dataKey="predicted" fill="#38f4b0" />
                      </BarChart>
                    </ResponsiveContainer>
                  </article>

                  <article className="panel">
                    <div className="panel-head">
                      <h3>Precision, Recall, F1</h3>
                    </div>
                    {classMetricRows.length > 0 ? (
                      <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={classMetricRows}>
                          <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                          <XAxis dataKey="label" stroke={chartColors.axis} tick={{ fill: chartColors.text, fontSize: 12 }} />
                          <YAxis domain={[0, 1]} stroke={chartColors.axis} tick={{ fill: chartColors.text, fontSize: 12 }} />
                          <Tooltip />
                          <Legend />
                          <Bar dataKey="precision" fill="#00d1ff" />
                          <Bar dataKey="recall" fill="#78ff9f" />
                          <Bar dataKey="f1" fill="#ffb347" />
                        </BarChart>
                      </ResponsiveContainer>
                    ) : (
                      <p>No class metric data yet. Train models again.</p>
                    )}
                  </article>

                  <article className="panel">
                    <div className="panel-head">
                      <h3>Top Classification Features</h3>
                    </div>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={trainInfo.classification.top_features} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                        <XAxis type="number" stroke={chartColors.axis} tick={{ fill: chartColors.text, fontSize: 12 }} />
                        <YAxis
                          type="category"
                          dataKey="feature"
                          width={150}
                          stroke={chartColors.axis}
                          tick={{ fill: chartColors.text, fontSize: 11 }}
                        />
                        <Tooltip />
                        <Bar dataKey="importance" fill="#ff7b7b" />
                      </BarChart>
                    </ResponsiveContainer>
                  </article>
                </section>

                <section className="panel">
                  <div className="panel-head">
                    <h3>Classification Prediction Form</h3>
                    <p>Submit a synthetic customer profile to get class prediction.</p>
                  </div>
                  <div className="predict-grid">
                    {trainInfo.classification.form_schema.map((field) => (
                      <div key={field.name} className="input-row">
                        <label>{field.name}</label>
                        {field.type === 'number' ? (
                          <input
                            type="number"
                            value={classInput[field.name] ?? ''}
                            onChange={(e) =>
                              setClassInput((prev) => ({ ...prev, [field.name]: Number(e.target.value) }))
                            }
                          />
                        ) : (
                          <select
                            value={classInput[field.name] ?? field.default}
                            onChange={(e) => setClassInput((prev) => ({ ...prev, [field.name]: e.target.value }))}
                          >
                            {field.options.map((op) => (
                              <option key={op} value={op}>{op}</option>
                            ))}
                          </select>
                        )}
                      </div>
                    ))}
                  </div>
                  <button className="btn-primary" onClick={predictClassification}>Predict Class</button>
                  {classPrediction && <p className="prediction">{classPrediction}</p>}
                </section>
              </>
            )}

            {showRegression && (
              <>
                <h2 className="section-title">Regression Analytics</h2>
                <section className="kpi-grid">
                  <article className="kpi-card">
                    <p>Mean Absolute Error</p>
                    <h3>{trainInfo.regression.mae.toFixed(3)}</h3>
                  </article>
                  <article className="kpi-card">
                    <p>R2 Score</p>
                    <h3>{trainInfo.regression.r2.toFixed(3)}</h3>
                  </article>
                  <article className="kpi-card">
                    <p>Model</p>
                    <h3>{formatModelName(trainInfo.regression.model_name)}</h3>
                  </article>
                </section>
                {trainInfo.regression.model_comparison?.length > 1 && (
                  <section className="panel">
                    <div className="panel-head">
                      <h3>Regression Model Comparison</h3>
                      <p>Sorted by R2 (higher is better), with MAE as tie-breaker.</p>
                    </div>
                    <div className="table-wrap">
                      <table className="preview-table">
                        <thead>
                          <tr>
                            <th>Model</th>
                            <th>R2</th>
                            <th>MAE</th>
                          </tr>
                        </thead>
                        <tbody>
                          {trainInfo.regression.model_comparison.map((row) => (
                            <tr key={row.model_name}>
                              <td>{formatModelName(row.model_name)}</td>
                              <td>{Number(row.r2).toFixed(4)}</td>
                              <td>{Number(row.mae).toFixed(4)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </section>
                )}

                <section className="panel">
                  <div className="panel-head">
                    <h3>Regression Prediction Form</h3>
                    <p>Enter feature values to get a numeric forecast.</p>
                  </div>
                  <div className="predict-grid">
                    {trainInfo.regression.form_schema.map((field) => (
                      <div key={field.name} className="input-row">
                        <label>{field.name}</label>
                        {field.type === 'number' ? (
                          <input
                            type="number"
                            value={regInput[field.name] ?? ''}
                            onChange={(e) => setRegInput((prev) => ({ ...prev, [field.name]: Number(e.target.value) }))}
                          />
                        ) : (
                          <select
                            value={regInput[field.name] ?? field.default}
                            onChange={(e) => setRegInput((prev) => ({ ...prev, [field.name]: e.target.value }))}
                          >
                            {field.options.map((op) => (
                              <option key={op} value={op}>{op}</option>
                            ))}
                          </select>
                        )}
                      </div>
                    ))}
                  </div>
                  <button className="btn-primary" onClick={predictRegression}>Predict Value</button>
                  {regPrediction && <p className="prediction">{regPrediction}</p>}
                </section>

                {trainInfo.regression.points?.length > 0 ? (
                  <section className="grid charts-grid">
                    <article className="panel">
                      <div className="panel-head">
                        <h3>Actual vs Predicted Scatter</h3>
                      </div>
                      <ResponsiveContainer width="100%" height={300}>
                        <ScatterChart>
                          <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                          <XAxis dataKey="actual" name="Actual" stroke={chartColors.axis} tick={{ fill: chartColors.text, fontSize: 12 }} />
                          <YAxis dataKey="predicted" name="Predicted" stroke={chartColors.axis} tick={{ fill: chartColors.text, fontSize: 12 }} />
                          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                          <Scatter data={trainInfo.regression.points} fill="#00d1ff" />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </article>

                    <article className="panel">
                      <div className="panel-head">
                        <h3>Actual vs Predicted Trend</h3>
                      </div>
                      <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={trainInfo.regression.points}>
                          <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                          <XAxis dataKey="sample" stroke={chartColors.axis} tick={{ fill: chartColors.text, fontSize: 12 }} />
                          <YAxis stroke={chartColors.axis} tick={{ fill: chartColors.text, fontSize: 12 }} />
                          <Tooltip />
                          <Legend />
                          <Line type="monotone" dataKey="actual" stroke="#38f4b0" dot={false} />
                          <Line type="monotone" dataKey="predicted" stroke="#ff7b7b" dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </article>

                    <article className="panel">
                      <div className="panel-head">
                        <h3>Error Distribution</h3>
                      </div>
                      <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={errorBins}>
                          <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                          <XAxis dataKey="range" hide />
                          <YAxis stroke={chartColors.axis} tick={{ fill: chartColors.text, fontSize: 12 }} />
                          <Tooltip />
                          <Bar dataKey="count" fill="#ffb347" />
                        </BarChart>
                      </ResponsiveContainer>
                    </article>
                  </section>
                ) : (
                  <section className="panel">
                    <h3>Regression Charts</h3>
                    <p>No regression plot data returned from backend.</p>
                  </section>
                )}

                <section className="grid charts-grid">
                  <article className="panel">
                    <div className="panel-head">
                      <h3>Top Regression Features</h3>
                    </div>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={trainInfo.regression.top_features} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                        <XAxis type="number" stroke={chartColors.axis} tick={{ fill: chartColors.text, fontSize: 12 }} />
                        <YAxis
                          type="category"
                          dataKey="feature"
                          width={150}
                          stroke={chartColors.axis}
                          tick={{ fill: chartColors.text, fontSize: 11 }}
                        />
                        <Tooltip />
                        <Bar dataKey="importance" fill="#38f4b0" />
                      </BarChart>
                    </ResponsiveContainer>
                  </article>
                </section>
              </>
            )}

            <h2 className="section-title">Executive Summary</h2>
            <section className="executive-layout">
              {showClassification && (
                <article className="panel executive-class-distribution">
                  <div className="panel-head">
                    <h3>Class Distribution</h3>
                  </div>
                  <div className="executive-class-chart">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={classDistributionRows}>
                        <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                        <XAxis dataKey="label" stroke={chartColors.axis} tick={{ fill: chartColors.text, fontSize: 12 }} />
                        <YAxis stroke={chartColors.axis} tick={{ fill: chartColors.text, fontSize: 12 }} />
                        <Tooltip />
                        <Bar dataKey="count">
                          {classDistributionRows.map((entry, idx) => (
                            <Cell key={`class-cell-${idx}`} fill={classColor(entry.label)} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </article>
              )}

              <article className="panel executive-correlation">
                <div className="panel-head">
                  <h3>Correlation Snapshot</h3>
                </div>
                <div className="heat-grid">
                  {trainInfo.summary.correlation.slice(0, 60).map((entry, idx) => (
                    <HeatCell key={`${entry.x}-${entry.y}-${idx}`} entry={entry} />
                  ))}
                </div>
              </article>

              {executiveNarrative && (
                <section className="panel narrative-panel executive-narrative">
                  <div className="panel-head">
                    <h3>Narrative Executive Summary</h3>
                    <p>Auto-generated interpretation of current model outputs for leadership decisions.</p>
                  </div>
                  <div className="narrative-grid">
                    <article className="narrative-card span-full">
                      <h4>Executive Narrative</h4>
                      <p>{executiveNarrative.narrative}</p>
                    </article>
                    <article className="narrative-card">
                      <h4>Strengths</h4>
                      {executiveNarrative.strengths.map((line, idx) => (
                        <p key={`strength-${idx}`}>- {line}</p>
                      ))}
                    </article>
                    <article className="narrative-card">
                      <h4>Weaknesses</h4>
                      {executiveNarrative.weaknesses.map((line, idx) => (
                        <p key={`weak-${idx}`}>- {line}</p>
                      ))}
                    </article>
                    <article className="narrative-card">
                      <h4>Suggestions</h4>
                      {executiveNarrative.suggestions.map((line, idx) => (
                        <p key={`sugg-${idx}`}>- {line}</p>
                      ))}
                    </article>
                    <article className="narrative-card">
                      <h4>Areas for Development</h4>
                      {executiveNarrative.development.map((line, idx) => (
                        <p key={`dev-${idx}`}>- {line}</p>
                      ))}
                    </article>
                    <article className="narrative-card">
                      <h4>Executive Growth Actions</h4>
                      {executiveNarrative.growthActions.map((line, idx) => (
                        <p key={`growth-${idx}`}>- {line}</p>
                      ))}
                    </article>
                    <article className="narrative-card">
                      <h4>Business KPIs</h4>
                      {executiveNarrative.kpis.map((line, idx) => (
                        <p key={`kpi-${idx}`}>- {line}</p>
                      ))}
                    </article>
                    <article className="narrative-card">
                      <h4>Model KPIs</h4>
                      {executiveNarrative.modelKpis.map((line, idx) => (
                        <p key={`mkpi-${idx}`}>- {line}</p>
                      ))}
                    </article>
                  </div>
                </section>
              )}
            </section>
          </div>
        )}
      </div>
    </div>
  )
}
