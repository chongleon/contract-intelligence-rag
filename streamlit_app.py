import html
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from llm_extract import llm_extract
    from rag_pipeline import rag_pipeline as rag_search
    from standardize import standardize, compare_policy, generate_compare_table
    from llm_explain import llm_explain

    def structured_pipeline(text: str) -> dict:
        result, _ = llm_extract(text)
        return result
    
    def rag_pipeline(query: str) -> dict:
        return rag_search(query)
except ImportError as e:
    st.error(f"导入失败: {e}")
    st.stop()


CITATION_PATTERN = re.compile(r"\[\[\s*(?:chunk_id\s*:\s*)?([^\[\]]+?)\s*\]\]")
# 注意：此处的值应与`rag_pipeline.py`中的`CHUNK_SIZE`的值保持一致
MAX_TOOLTIP_CHARS = 600


def render_answer_with_citations(answer, contexts) -> str:
    """将 [[chunk_id]] 标注渲染为可悬浮提示的引用，同时展示来源 chips。"""

    if not answer:
        return ""

    context_map = {}
    for ctx in contexts:
        if not isinstance(ctx, dict):
            continue
        chunk_id = ctx.get("chunk_id")
        if chunk_id:
            context_map[str(chunk_id).strip()] = ctx

    parts = []
    last_idx = 0
    for match in CITATION_PATTERN.finditer(answer):
        parts.append(html.escape(answer[last_idx:match.start()]))
        chunk_id = match.group(1).strip()
        ctx = context_map.get(chunk_id)
        raw_tooltip = (ctx.get("content") if ctx else "") or ""
        tooltip = raw_tooltip.strip().replace("\n", " ")
        if len(tooltip) > MAX_TOOLTIP_CHARS:
            tooltip = tooltip[:MAX_TOOLTIP_CHARS] + "..."
        tooltip = html.escape(tooltip)
        label_text = ctx.get("source") if ctx else chunk_id
        label = html.escape(str(label_text or chunk_id))
        data_chunk = html.escape(chunk_id)
        parts.append(
            f"<sup class='citation' title='{tooltip}' data-chunk='{data_chunk}'>[{label}]</sup>"
        )
        last_idx = match.end()
    parts.append(html.escape(answer[last_idx:]))

    text_html = "".join(parts).replace("\n", "<br>")

    chip_html = []
    for idx, ctx in enumerate(contexts, start=1):
        if not isinstance(ctx, dict):
            continue
        chunk_id = str(ctx.get("chunk_id") or f"chunk_{idx}")
        source = str(ctx.get("source") or "")
        label = source or chunk_id
        tooltip = (ctx.get("content") or "").strip().replace("\n", " ")
        if len(tooltip) > MAX_TOOLTIP_CHARS:
            tooltip = tooltip[:MAX_TOOLTIP_CHARS] + "..."
        chip_html.append(
            f"<span class='source-chip' title='{html.escape(tooltip)}'>"
            f"{html.escape(label)}</span>"
        )

    chips_section = (
        "<div class='source-chips'>" + "".join(chip_html) + "</div>"
        if chip_html else ""
    )

    return f"<div class='answer-block'><div class='answer-text'>{text_html}</div>{chips_section}</div>"

# 页面
st.set_page_config(
    page_title="保险条款解读与对比工具",
    page_icon="📋",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #1E3D59;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.8rem;
        color: #1E3D59;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    .metric-label {
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    .metric-value {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
    }
    .info-box {
        background-color: #f0f7fb;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E3D59;
        margin-bottom: 1rem;
    }
    .explain-box {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .exclusion-box {
        background-color: #fff3cd;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 5px solid #ffc107;
        max-height: 300px;
        overflow-y: auto;
        font-size: 0.95rem;
    }
    .exclusion-item {
        padding: 0.3rem 0;
        border-bottom: 1px solid #ffe69c;
    }
    .exclusion-item:last-child {
        border-bottom: none;
    }
    .source-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.3rem;
    }
    .source-chip {
        background-color: #eef2f7;
        color: #1E3D59;
        padding: 0.2rem 0.8rem;
        border-radius: 999px;
        font-size: 0.85rem;
        border: 1px solid #d0d8e8;
        cursor: help;
        max-width: 320px;
        text-overflow: ellipsis;
        white-space: nowrap;
        overflow: hidden;
    }
    .source-chip:hover {
        background-color: #dfe8f4;
    }
    .answer-block {
        line-height: 1.6;
    }
    .answer-text {
        font-size: 1rem;
        color: #1E3D59;
    }
    .citation {
        background-color: #e6eef7;
        color: #1E3D59;
        border-radius: 4px;
        padding: 0 0.2rem;
        margin-left: 0.2rem;
        font-size: 0.75rem;
        cursor: help;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">保险条款智能解读工具</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 功能导航")
    function = st.radio(
        "选择功能：",
        ["单条款解读", "产品对比", "智能问答"]
    )

# 单条款解读
if function == "单条款解读":
    st.markdown('<p class="section-title">📄 单条款通俗化解读</p>', unsafe_allow_html=True)
    
    if "last_uploaded" not in st.session_state:
        st.session_state.last_uploaded = None
    
    col1, col2 = st.columns([1, 1.8])
    
    with col1:
        st.markdown("### 上传条款文件")
        
        uploaded_file = st.file_uploader(
            "选择TXT格式的保险条款文件", 
            type=['txt'], 
            key="single_upload",
            help="请上传保险条款的TXT文件"
        )
        
        if uploaded_file is not None:
            st.session_state.last_uploaded = uploaded_file
        
        analyze_btn = st.button("开始解读", type="primary", use_container_width=True, disabled=uploaded_file is None)
    
    with col2:
        st.markdown("### 解读结果")
    
        if analyze_btn:
            with st.spinner("正在分析..."):
                if st.session_state.last_uploaded is not None:
                    content = st.session_state.last_uploaded.read().decode('utf-8', errors='ignore')
                    st.session_state.last_uploaded.seek(0)
                    raw_data = structured_pipeline(content)
                    st.success(f"已上传: {st.session_state.last_uploaded.name}")
                else:
                    st.warning("请上传文件")
                    st.stop()
            
                std_data = standardize(raw_data)
            
                col_left, col_right = st.columns(2)
            
                with col_left:
                # 产品名称
                    product_name = std_data.get("product_name", "")
                    if product_name and product_name != "-":
                        st.markdown('<p class="metric-label">产品名称</p>', unsafe_allow_html=True)
                        st.markdown(f'<p class="metric-value">{product_name}</p>', unsafe_allow_html=True)
                
                # 保险类型
                    insurance_type = std_data.get("insurance_type", "")
                    if insurance_type and insurance_type != "-":
                        st.markdown('<p class="metric-label">保险类型</p>', unsafe_allow_html=True)
                        st.markdown(f'<p class="metric-value">{insurance_type}</p>', unsafe_allow_html=True)
                
                # 保障期限
                    coverage_period = std_data.get("coverage_period_raw", "")
                    if coverage_period and coverage_period != "-":
                        st.markdown('<p class="metric-label">保障期限</p>', unsafe_allow_html=True)
                        st.markdown(f'<p class="metric-value">{coverage_period}</p>', unsafe_allow_html=True)
            
                with col_right:
                # 保额
                    amount_val = std_data.get("coverage_amount_value")
                    if amount_val:
                        if amount_val >= 100000000:
                            amount_display = f"{amount_val/100000000:.0f}亿"
                        elif amount_val >= 10000:
                            amount_display = f"{amount_val/10000:.0f}万"
                        else:
                            amount_display = f"{amount_val:.0f}元"
                    else:
                        amount_display = std_data.get("coverage_amount_raw", "")
                
                    if amount_display and amount_display != "-":
                        st.markdown('<p class="metric-label">保额</p>', unsafe_allow_html=True)
                        st.markdown(f'<p class="metric-value">{amount_display}</p>', unsafe_allow_html=True)
                
                # 等待期
                    waiting_period = std_data.get("waiting_period_raw", "")
                    if waiting_period and waiting_period != "-":
                        st.markdown('<p class="metric-label">等待期</p>', unsafe_allow_html=True)
                        st.markdown(f'<p class="metric-value">{waiting_period}</p>', unsafe_allow_html=True)
            
                st.markdown("---")
            
            # 免责条款
                exclusions = std_data.get("exclusions", [])
                if exclusions:
                    st.markdown("### ⚠️ 免责条款")
                    exclusion_html = '<div class="exclusion-box">'
                    for excl in exclusions:
                        exclusion_html += f'<div class="exclusion-item">• {excl}</div>'
                    exclusion_html += '</div>'
                    st.markdown(exclusion_html, unsafe_allow_html=True)
            
            # 通俗化解释
                with st.spinner("正在生成通俗化解释..."):
                    explain_result = llm_explain(raw_data)
                
                    has_explain = any([
                        explain_result.get("waiting_period_explanation"),
                        explain_result.get("coverage_explanation"),
                        explain_result.get("suitable_for"),
                        explain_result.get("risk_warning")
                    ])
                
                    if has_explain:
                        st.markdown("### 💡 通俗化解释")
                    
                        explain_html = '<div class="explain-box">'
                    
                        if explain_result.get("waiting_period_explanation"):
                            explain_html += f'<p><strong>⏱️ 等待期：</strong>{explain_result["waiting_period_explanation"]}</p>'
                    
                        if explain_result.get("coverage_explanation"):
                            explain_html += f'<p><strong>🛡️ 保障范围：</strong>{explain_result["coverage_explanation"]}</p>'
                    
                        if explain_result.get("suitable_for"):
                            explain_html += f'<p><strong>👥 适合人群：</strong>{explain_result["suitable_for"]}</p>'
                    
                        if explain_result.get("risk_warning"):
                            explain_html += f'<p><strong>⚠️ 风险提示：</strong>{explain_result["risk_warning"]}</p>'
                    
                        explain_html += '</div>'
                        st.markdown(explain_html, unsafe_allow_html=True)
            
                with st.expander("查看结构化数据"):
                    st.json(std_data)

# 多产品对比
elif function == "产品对比":
    st.markdown('<p class="section-title">两款产品对比分析</p>', unsafe_allow_html=True)

    if "last_upload_a" not in st.session_state:
        st.session_state.last_upload_a = None
    if "last_upload_b" not in st.session_state:
        st.session_state.last_upload_b = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 产品A")
        
        upload_a = st.file_uploader("上传产品A的TXT文件", type=['txt'], key="a")
        
        if upload_a is not None:
            st.session_state.last_upload_a = upload_a
    
    with col2:
        st.markdown("### 产品B")
        
        upload_b = st.file_uploader("上传产品B的TXT文件", type=['txt'], key="b")
        
        if upload_b is not None:
            st.session_state.last_upload_b = upload_b
    
    compare_btn = st.button("开始对比", type="primary", use_container_width=True)
    
    if compare_btn:
        with st.spinner("正在对比分析..."):
            if st.session_state.last_upload_a is not None:
                text_a = st.session_state.last_upload_a.read().decode('utf-8', errors='ignore')
                st.session_state.last_upload_a.seek(0)
                raw_a = structured_pipeline(text_a)
                name_a = st.session_state.last_upload_a.name
            else:
                st.warning("请上传产品A的文件")
                st.stop()
            
            if st.session_state.last_upload_b is not None:
                text_b = st.session_state.last_upload_b.read().decode('utf-8', errors='ignore')
                st.session_state.last_upload_b.seek(0)
                raw_b = structured_pipeline(text_b)
                name_b = st.session_state.last_upload_b.name
            else:
                st.warning("请上传产品B的文件")
                st.stop()
            
            std_a = standardize(raw_a)
            std_b = standardize(raw_b)
            
            raw_table = generate_compare_table(std_a, std_b, name_a, name_b)
            
            # 过滤掉两个都为空的行
            if not raw_table.empty:
                value_cols = [col for col in raw_table.columns if col != "对比项"]
                
                rows_to_keep = []
                for idx, row in raw_table.iterrows():
                    has_value = False
                    for col in value_cols:
                        val = row[col]
                        if val and val != "-" and val != "None" and str(val).strip():
                            has_value = True
                            break
                    rows_to_keep.append(has_value)
                
                filtered_table = raw_table[rows_to_keep].reset_index(drop=True)
            else:
                filtered_table = raw_table
        
        st.success("对比完成！")
        
        if not filtered_table.empty:
            st.dataframe(filtered_table, use_container_width=True, hide_index=True)
        else:
            st.info("没有可显示的对比数据")
        
        with st.expander("查看详细数据"):
            col1, col2 = st.columns(2)
            with col1:
                st.json(std_a)
            with col2:
                st.json(std_b)

# 智能问答
else:
    st.markdown('<p class="section-title">智能问答</p>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    with st.sidebar:
        st.markdown("### 示例问题")
        examples = [
            "请求赔偿时，应提交哪些索赔材料？",
            "哪些情况不赔？",
            "等待期是什么？",
            "免责条款有哪些？"
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True):
                st.session_state["example_query"] = ex
    
    # 显示聊天记录
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("html_content"):
                st.markdown(msg["html_content"], unsafe_allow_html=True)
            else:
                st.write(msg["content"])
                if msg["role"] == "assistant" and msg.get("source"):
                    st.caption(f"来源: {msg['source']}")
    
    # 处理示例问题
    if "example_query" in st.session_state:
        query = st.session_state.pop("example_query")
    else:
        query = None
    
    # 输入框
    prompt = st.chat_input("请输入您的问题...")
    
    if prompt or query:
        user_input = prompt or query
        
        # 显示用户问题并保存
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 调用RAG获取回答
        with st.chat_message("assistant"):
            with st.spinner("正在检索..."):
                rag_result = rag_pipeline(user_input)

            answer = (rag_result.get("answer") or "").strip()
            contexts = rag_result.get("contexts") or []

            if not answer:
                answer = "未找到相关条款，暂无法回答。"

            rendered_answer = render_answer_with_citations(answer, contexts)
            if rendered_answer:
                st.markdown(rendered_answer, unsafe_allow_html=True)
            else:
                st.write(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "html_content": rendered_answer,
                "contexts": contexts,
            })

# 页脚
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: gray; font-size: 0.8rem;'>
        <p>保险条款解读与对比工具</p>
    </div>
    """,
    unsafe_allow_html=True
)