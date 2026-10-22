import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='LLM Benchmark Dashboard', layout='wide')
st.title('LLM Benchmark Dashboard')
st.caption('JSON/CSV 벤치마크 결과 탐색 · 동시 요청 수별 비교')

DEFAULT_ROOT = str(Path(__file__).resolve().parent / 'vllm-result')
root_text = st.sidebar.text_input('결과 디렉터리', DEFAULT_ROOT)
root = Path(root_text).expanduser()
if not root.is_dir():
    st.error(f'디렉터리가 존재하지 않는다: {root}')
    st.stop()

files = sorted([p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.json', '.csv'}])
if not files:
    st.warning('JSON 또는 CSV 파일이 없다.')
    st.stop()

selected_runs = st.sidebar.multiselect('실험 폴더', sorted({str(p.relative_to(root).parts[0]) for p in files}), default=sorted({str(p.relative_to(root).parts[0]) for p in files}))
files = [p for p in files if p.relative_to(root).parts[0] in selected_runs]


def flatten_obj(obj, prefix=''):
    result = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f'{prefix}.{k}' if prefix else str(k)
            result.update(flatten_obj(v, key))
    elif isinstance(obj, list):
        # 긴 요청별 결과는 별도 원본 뷰에서 확인하고, 요약표에는 크기를 남긴다.
        result[f'{prefix}.__count'] = len(obj)
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        result[prefix] = obj
    return result


def metadata(path):
    parts = path.relative_to(root).parts
    run_name = parts[0]
    config_name = next((part for part in parts[1:-1] if re.match(r'^\d+\.\d+\.\d+$', part)), '')
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)$', config_name)
    in_tok, out_tok, conc = (int(x) for x in match.groups()) if match else (None, None, None)
    return dict(run=run_name, config=config_name, concurrency=conc, input_tokens_config=in_tok,
                output_tokens_config=out_tok, source=path.suffix.lower()[1:], file=str(path.relative_to(root)))


rows, issues = [], []
for path in files:
    meta = metadata(path)
    try:
        if path.suffix.lower() == '.json':
            with path.open('r', encoding='utf-8') as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                row = {**meta, **flatten_obj(data)}
            elif isinstance(data, list):
                row = {**meta, 'root_list.__count': len(data)}
            else:
                row = {**meta, 'root_value': data}
            rows.append(row)
        else:
            df = pd.read_csv(path)
            row = {**meta, 'csv_rows': len(df), 'csv_columns': len(df.columns)}
            if len(df) == 1:
                row.update({f'csv.{k}': v for k, v in df.iloc[0].to_dict().items()})
            else:
                # 행별 메트릭은 평균을 산출하되 정확한 해석은 컬럼 의미 확인 필요.
                for col in df.select_dtypes(include='number').columns:
                    row[f'csv.mean.{col}'] = df[col].mean()
            rows.append(row)
    except Exception as exc:
        issues.append(f'{path.relative_to(root)}: {exc}')

if issues:
    with st.expander(f'읽기 오류 {len(issues)}건'):
        st.code('\n'.join(issues))
if not rows:
    st.warning('읽을 수 있는 결과가 없다.')
    st.stop()

full = pd.DataFrame(rows)
fixed_cols = ['run', 'config', 'concurrency', 'input_tokens_config', 'output_tokens_config', 'source', 'file']
metrics = []
for col in full.columns:
    if col in fixed_cols or col.endswith('.__count') or col in ('csv_rows', 'csv_columns'):
        continue
    numeric = pd.to_numeric(full[col], errors='coerce')
    if numeric.notna().any() and not pd.api.types.is_bool_dtype(full[col]):
        full[col] = numeric
        metrics.append(col)

left, middle, right = st.columns(3)
left.metric('실험 폴더', full['run'].nunique())
middle.metric('결과 파일', len(full))
right.metric('수치형 측정 항목', len(metrics))

st.subheader('실험 결과 표')
st.dataframe(full, use_container_width=True, hide_index=True)

st.subheader('동시 요청 수별 성능 비교')
if not metrics or full['concurrency'].isna().all():
    st.info('수치형 필드 또는 폴더 이름의 concurrency를 찾지 못했다. 아래 원본 데이터를 확인하면 된다.')
else:
    suggested = next((c for c in metrics if any(term in c.lower() for term in ('throughput', 'ttft', 'tpot'))), metrics[0])
    metric = st.selectbox('비교할 측정 항목', metrics, index=metrics.index(suggested))
    sources = st.multiselect('결과 형식', sorted(full['source'].unique()), default=sorted(full['source'].unique()))
    frame = full[full['source'].isin(sources)].dropna(subset=['concurrency', metric]).copy()
    if not frame.empty:
        frame['experiment'] = frame['run'] + ' / ' + frame['source']
        # 같은 설정 반복 측정은 평균으로 표시한다.
        grouped = frame.groupby(['experiment', 'concurrency'], as_index=False)[metric].mean()
        fig = px.line(grouped, x='concurrency', y=metric, color='experiment', markers=True,
                      title=f'{metric} / 동시 요청 수')
        fig.update_layout(xaxis_title='Concurrency', yaxis_title=metric)
        st.plotly_chart(fig, use_container_width=True)
        st.caption('동일한 실험·동시 요청 수에서 중복 파일이 존재하면 평균을 표시한다. 단위와 지표 정의는 원본 포맷에서 확인해야 한다.')
    else:
        st.info('해당 항목에 유효한 데이터가 없다.')

st.subheader('원본 결과 확인')
choice = st.selectbox('파일 선택', [str(p.relative_to(root)) for p in files])
chosen = root / choice
if chosen.suffix.lower() == '.json':
    with chosen.open('r', encoding='utf-8') as handle:
        st.json(json.load(handle), expanded=False)
else:
    raw = pd.read_csv(chosen)
    st.dataframe(raw, use_container_width=True, hide_index=True)
    st.caption(f'총 {len(raw):,}행')

st.download_button('현재 비교표 CSV 저장', data=full.to_csv(index=False).encode('utf-8-sig'),
                   file_name='benchmark_summary.csv', mime='text/csv')
