import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import structured_pipeline, rag_pipeline
    from standardize import standardize, compare_policy, generate_compare_table
    from llm_explain import llm_explain

    REAL_MODULES = True

    PRODUCT_LIST = {
        "重疾险 - XX重疾险（50万）": {
            "product_name": "XX重疾险（2024版）",
            "insurance_type": "重疾险",
            "coverage_period": "终身",
            "coverage_amount": "50万元",
            "waiting_period": "90天",
            "exclusions": ["故意杀害", "犯罪", "自杀", "战争", "核辐射"]
        },
        "医疗险 - YY医疗险（100万）": {
            "product_name": "YY医疗险（基础版）",
            "insurance_type": "医疗险",
            "coverage_period": "1年",
            "coverage_amount": "100万元",
            "waiting_period": "30天",
            "exclusions": ["既往症", "美容手术", "牙科治疗"]
        },
        "重疾险 - ZZ重疾险（30万）": {
            "product_name": "ZZ重疾险（青春版）",
            "insurance_type": "重疾险",
            "coverage_period": "至70岁",
            "coverage_amount": "30万元",
            "waiting_period": "60天",
            "exclusions": ["酒驾", "高风险运动"]
        }
    }
    
except ImportError as e:
    REAL_MODULES = False

    PRODUCT_LIST = {
        "重疾险 - XX重疾险（50万）": {
            "product_name": "XX重疾险（2024版）",
            "insurance_type": "重疾险",
            "coverage_period": "终身",
            "coverage_amount": "50万元",
            "waiting_period": "90天",
            "exclusions": ["故意杀害", "犯罪", "自杀", "战争", "核辐射"]
        },
        "医疗险 - YY医疗险（100万）": {
            "product_name": "YY医疗险（基础版）",
            "insurance_type": "医疗险",
            "coverage_period": "1年",
            "coverage_amount": "100万元",
            "waiting_period": "30天",
            "exclusions": ["既往症", "美容手术", "牙科治疗"]
        },
        "重疾险 - ZZ重疾险（30万）": {
            "product_name": "ZZ重疾险（青春版）",
            "insurance_type": "重疾险",
            "coverage_period": "至70岁",
            "coverage_amount": "30万元",
            "waiting_period": "60天",
            "exclusions": ["酒驾", "高风险运动"]
        }
    }
    
    def structured_pipeline(text: str) -> dict:
        import re
        result = {
            "product_name": "示例保险",
            "insurance_type": "重疾险" if "重疾" in text else "医疗险",
            "coverage_period": "终身" if "终身" in text else "1年",
            "coverage_amount": "50万元",
            "waiting_period": "90天",
            "exclusions": ["故意杀害", "犯罪", "自杀"]
        }

        amount_match = re.search(r'(\d+)[万万元]', text)
        if amount_match:
            result["coverage_amount"] = amount_match.group(0)
        return result
    
    def rag_pipeline(query: str) -> list:
        rag_data = {
            "等待期": [{
                "content": "等待期是指保险合同生效后的一段时间内,如果发生保险事故,保险公司不承担赔偿责任。重疾险通常为90天,医疗险通常为30天。",
                "score": 0.89,
                "source": "covid.pdf",
                "chunk_id": "covid_3"
            }],
            "不赔": [{
                "content": "免责条款包括:投保人故意杀害、被保险人犯罪、自杀、战争、核辐射、既往症、酒驾等情形。",
                "score": 0.92,
                "source": "exclusions.pdf",
                "chunk_id": "exclusions_1"
            }],
            "理赔": [{
                "content": "理赔流程:1. 及时报案 2. 准备理赔材料(病历、诊断证明等)3. 提交申请 4. 保险公司审核 5. 赔付。",
                "score": 0.85,
                "source": "claims.pdf",
                "chunk_id": "claims_2"
            }],
            "免责": [{
                "content": "责任免除条款:因下列情形之一导致被保险人发生保险事故的,保险公司不承担给付保险金责任:1. 投保人故意杀害;2. 被保险人犯罪;3. 自杀;4. 战争;5. 核辐射;6. 既往症;7. 酒驾。",
                "score": 0.91,
                "source": "exclusions.pdf",
                "chunk_id": "exclusions_2"
            }]
        }
        for key in rag_data:
            if key in query:
                return rag_data[key]
        return [{"content": f"关于'{query}'的问题，建议查看具体条款或咨询保险公司。", "score": 0.50}]
    
    def standardize(data: dict) -> dict:
        from standardize import standardize as real_std
        return real_std(data)
    
    def compare_policy(a: dict, b: dict) -> dict:
        from standardize import compare_policy as real_cmp
        return real_cmp(a, b)
    
    def generate_compare_table(a: dict, b: dict, name_a="产品A", name_b="产品B") -> pd.DataFrame:
        from standardize import generate_compare_table as real_tbl
        return real_tbl(a, b, name_a, name_b)
    


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
        font-size: 1.5rem;
        color: #1E3D59;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f7fb;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E3D59;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">保险条款智能解读工具</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 功能导航")
    function = st.radio(
        "选择功能：",
        ["单条款解读", "多产品对比", "智能问答"]
    )
    
    if not REAL_MODULES:
        st.info("演示模式（使用模拟数据）")

# 单条款解读
if function == "单条款解读":
    st.markdown('<p class="section-title">单条款通俗化解读</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("### 1. 选择条款")
        uploaded_file = st.file_uploader("上传条款文件（TXT格式）", type=['txt'])
        
        st.markdown("### 或选择示例")
        product_names = ["请选择..."] + list(PRODUCT_LIST.keys())
        selected = st.selectbox("选择示例产品", product_names)
        
        analyze_btn = st.button("开始解读", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 2. 解读结果")
        
        if analyze_btn:
            if uploaded_file:
                content = uploaded_file.read().decode('utf-8', errors='ignore')
                raw_data = structured_pipeline(content)
                st.success(f"已上传: {uploaded_file.name}")
            elif selected != "请选择...":
                raw_data = PRODUCT_LIST[selected]
                st.success(f"已选择: {selected}")
            else:
                st.warning("请上传文件或选择示例")
                st.stop()
            
            std_data = standardize(raw_data)
            
            with st.spinner("正在分析..."):
                col_left, col_right = st.columns(2)
                with col_left:
                    st.metric("产品名称", std_data.get("product_name", "-"))
                    st.metric("保险类型", std_data.get("insurance_type", "-"))
                    st.metric("保障期限", std_data.get("coverage_period_raw", "-"))
                with col_right:
                    st.metric("保额", std_data.get("coverage_amount_raw", "-"))
                    st.metric("等待期", std_data.get("waiting_period_raw", "-"))
            
            st.subheader("免责条款")
            exclusions = std_data.get("exclusions", [])
            if exclusions:
                exclusions_text = "、".join(exclusions)
                st.warning(f"• {exclusions_text}")
            else:
                st.info("无免责条款")
            
            with st.expander("查看结构化数据"):
                st.json(std_data)

# 多产品对比
elif function == "多产品对比":
    st.markdown('<p class="section-title">两款产品对比分析</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 产品A")
        upload_a = st.file_uploader("上传产品A", type=['txt'], key="a")
        preset_a = st.selectbox("或选择预设", ["请选择..."] + list(PRODUCT_LIST.keys()), key="pa")
    
    with col2:
        st.markdown("### 产品B")
        upload_b = st.file_uploader("上传产品B", type=['txt'], key="b")
        preset_b = st.selectbox("或选择预设", ["请选择..."] + list(PRODUCT_LIST.keys()), key="pb")
    
    compare_btn = st.button("开始对比", type="primary", use_container_width=True)
    
    if compare_btn:
        if upload_a:
            text_a = upload_a.read().decode('utf-8', errors='ignore')
            raw_a = structured_pipeline(text_a)
            name_a = upload_a.name
        elif preset_a != "请选择...":
            raw_a = PRODUCT_LIST[preset_a]
            name_a = preset_a
        else:
            st.warning("请选择产品A")
            st.stop()
        
        if upload_b:
            text_b = upload_b.read().decode('utf-8', errors='ignore')
            raw_b = structured_pipeline(text_b)
            name_b = upload_b.name
        elif preset_b != "请选择...":
            raw_b = PRODUCT_LIST[preset_b]
            name_b = preset_b
        else:
            st.warning("请选择产品B")
            st.stop()
        
        with st.spinner("正在对比分析..."):
            std_a = standardize(raw_a)
            std_b = standardize(raw_b)
            table = generate_compare_table(std_a, std_b, name_a, name_b)
        
        st.success("对比完成！")
        
        st.dataframe(table, use_container_width=True, hide_index=True)
        
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
        st.session_state.messages = [
            {"role": "assistant", "content": "您好！我是保险条款解读助手，有什么可以帮您的吗？"}
        ]
    
    with st.sidebar:
        st.markdown("### 示例问题")
        examples = ["等待期是什么？", "哪些情况不赔？", "怎么理赔？", "免责条款有哪些？"]
        for ex in examples:
            if st.button(ex, use_container_width=True):
                st.session_state["example_query"] = ex
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if "example_query" in st.session_state:
        query = st.session_state.pop("example_query")
    else:
        query = None
    
    prompt = st.chat_input("请输入您的问题...")
    
    if prompt or query:
        user_input = prompt or query
        
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("assistant"):
            with st.spinner("正在检索条款..."):
                results = rag_pipeline(user_input)
            
            if results:
                st.write(f"找到 {len(results)} 条相关条款：")
    
                if results and len(results) > 0:
                    first_result = results[0]
                    answer = first_result.get('content', '')
                    st.success("**回答：**")
                    st.write(answer)

            else:
                st.write("未找到相关条款。")
        
        st.session_state.messages.append(
            {"role": "assistant", "content": f"已检索到 {len(results) if results else 0} 条相关条款。"}
        )

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