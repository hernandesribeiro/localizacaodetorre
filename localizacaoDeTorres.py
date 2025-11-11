"""
Streamlit app: Localizador de Torres

Este arquivo cria um app Streamlit para localizar torres a partir de uma planilha (Excel ou CSV).
Funcionalidades:
- Permite upload de arquivo (Excel .xls/.xlsx ou CSV) via interface ou lê um arquivo local se existir
- Detecta automaticamente colunas de latitude/longitude (ou permite mapear colunas manualmente)
- Mostra tabela filtrável e resumo
- Exibe mapa interativo com PyDeck (aglomeração por clusters opcional)
- Permite baixar os dados filtrados

Instruções de uso:
1. Instale dependências: pip install -r requirements.txt
2. Rode: streamlit run streamlit_localizador_torres.py

Salve este arquivo no mesmo diretório que a planilha ou simplesmente abra o app e faça upload.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import tempfile
from collections import defaultdict
import os 

# --- CSS CUSTOMIZADO PARA NEGRITO NOS RÓTULOS (Simulando "negrito 14") ---
st.markdown("""
<style>
/* Aplica negrito e cor mais escura aos rótulos de todos os inputs (Selectbox e Number Input) */
label {
    font-weight: bold !important;
    color: #333333 !important; /* Cor mais escura para dar ênfase */
}
</style>
""", unsafe_allow_html=True)
# --------------------------------------------------------------------------

# ----------------------------------------------------------------------
# INÍCIO DO STREAMLIT
# ----------------------------------------------------------------------

st.set_page_config(page_title="Localizador de Torres ⚡", layout="wide")

st.title("⚡ Localizador de Torres — Versão Final")

# Upload do arquivo Excel
arquivo = st.file_uploader("📁 Selecione o arquivo Excel", type=["xlsx", "xls"])

if arquivo is not None:
    try:
        excel_file = pd.ExcelFile(arquivo)
        abas = excel_file.sheet_names

        # --- Lógica de Leitura DADOS, KM_LT (MANTIDA) ---
        df_dados = pd.read_excel(arquivo, sheet_name="DADOS").fillna("")
        
        if "CONCESSÕES" not in df_dados.columns or "LT" not in df_dados.columns:
            st.error("❌ A aba 'DADOS' deve conter exatamente as colunas 'CONCESSÕES' e 'LT'.")
            st.stop()
            
        df_dados["CONCESSÕES"] = df_dados["CONCESSÕES"].astype(str).str.strip()
        df_dados["LT"] = df_dados["LT"].astype(str).str.strip()

        # POPULA CONCESSÕES
        todas_concessoes = sorted(df_dados["CONCESSÕES"].unique().tolist())
        todas_concessoes = [c for c in todas_concessoes if c != ""]
        
        # CÁLCULO DO COMPRIMENTO (MANTIDO)
        comprimento = None
        df_km = pd.read_excel(arquivo, sheet_name="KM_LT").fillna("") if "KM_LT" in abas else pd.DataFrame()
        
        # --- CARREGAMENTO DO MAPA DE TORRES JBJU (MANTIDO) ---
        torres_jbju_map = {}
        if "Torres JBJU" in abas:
            df_jbju = pd.read_excel(arquivo, sheet_name="Torres JBJU").fillna("")
            
            if len(df_jbju.columns) >= 5: 
                df_jbju.columns = [str(c).strip().lower().replace(' ', '') for c in df_jbju.columns]
                
                codigo_col = df_jbju.columns[0]   
                figura_col = df_jbju.columns[1]   
                sequencia_col = df_jbju.columns[2]  
                imagem_col = df_jbju.columns[4]     
                
                torres_jbju_map = df_jbju.set_index(codigo_col).apply(
                    lambda row: (str(row[figura_col]).strip(), str(row[sequencia_col]).strip().upper(), str(row[imagem_col]).strip()), axis=1
                ).to_dict()
            else:
                st.warning("⚠️ A aba 'Torres JBJU' deve ter pelo menos 5 colunas para ler o Caminho da Imagem da COLUNA E. (A, B, C, D, E)")
        
        # --- LAYOUT DE INPUTS (AGRUPADO NA MESMA LINHA) ---
        # 5 COLUNAS PARA CONCESSÃO, LT, FASE, MÉTODO e KM DE BUSCA
        col_concessao, col_lt, col_fase, col_metodo, col_km_input = st.columns([1.5, 1.5, 0.8, 1.5, 1.5]) 

        with col_concessao:
            concessao_escolhida = st.selectbox("🔹 Concessão:", todas_concessoes)

        lt_escolhida = None
        if concessao_escolhida:
            df_filtrado_lt = df_dados[
                (df_dados["CONCESSÕES"] == concessao_escolhida) &
                (df_dados["LT"] != "")
            ]
            lts = sorted(df_filtrado_lt["LT"].unique().tolist())
            with col_lt:
                lt_escolhida = st.selectbox("🔹 LT:", lts) if lts else None

        with col_fase:
            fase_escolhida = st.selectbox("🔹 Fase Defeito:", ["A", "B", "C"])

        with col_metodo:
            metodo = st.selectbox(
                "⚙️ Método:",
                ["Sequência Negativa", "TW", "SIGRA 1 Terminal", "SIGRA 2 Terminais"]
            )
        
        with col_km_input:
            valor_busca = st.number_input(
                "🎯 KM de Busca:", 
                min_value=0.0,
                step=0.1,
                format="%.2f",
                value=0.0
            )

        km_calculado = valor_busca
        
        # --- MÉTRICA DE COMPRIMENTO (MANTIDA EM LINHA SEPARADA) ---
        st.markdown("---") 
        
        if lt_escolhida:
             if "KM_LT" in abas and "LT" in df_km.columns and "KM" in df_km.columns:
                df_km["LT"] = df_km["LT"].astype(str).str.strip()
                linha_lt = df_km[df_km["LT"] == str(lt_escolhida).strip()]
                if not linha_lt.empty:
                    try:
                        comprimento = pd.to_numeric(linha_lt["KM"].iloc[0])
                    except Exception:
                        comprimento = None
            
        # A métrica é separada porque é um OUTPUT e ficaria desalinhada com os inputs
        col_metrica, col_gap = st.columns([1, 5])
        with col_metrica:
            if comprimento is not None:
                st.metric(label="📏 Comprimento (km)", value=f"{comprimento:.2f}")
            else:
                st.warning("Comprimento N/D", icon="⚠️")
        
        st.markdown("---")


        # --- CONTINUAÇÃO DA LÓGICA (Gráfico e Tabela) ---
        if lt_escolhida:
            st.subheader("📈 Representação da Sequência de Fases")
            
            plotar_clicado = st.button("🔍 Plotar Resultados")

            torres_na_janela_df = None
            
            if plotar_clicado and lt_escolhida in abas and valor_busca > 0:
                
                df_lt = pd.read_excel(arquivo, sheet_name=lt_escolhida)
                df_lt.columns = [str(c).strip().lower().replace(' ', '') for c in df_lt.columns]

                km_col = "km"
                desc_col = "descriçãolocalização"
                fase_seq_col = "fases" 
                
                cols_ok = km_col in df_lt.columns and desc_col in df_lt.columns and fase_seq_col in df_lt.columns
                if not cols_ok:
                    st.error(f"❌ Colunas esperadas (KM, Descrição Localização, FASES) não encontradas na aba {lt_escolhida}.")
                    st.stop()

                df_lt = df_lt.dropna(subset=[km_col])
                df_lt[km_col] = pd.to_numeric(df_lt[km_col], errors="coerce")
                df_lt = df_lt.dropna(subset=[km_col]).sort_values(km_col).reset_index(drop=True)

                torre_idx = df_lt[df_lt[km_col] >= valor_busca].index
                
                if len(torre_idx) > 0:
                    idx_central = torre_idx[0]
                    
                    start_idx = max(0, idx_central - 2)
                    end_idx = min(len(df_lt) - 1, idx_central + 2)
                    
                    df_plot = df_lt.loc[start_idx:end_idx].copy()
                    df_plot["x_pos"] = np.linspace(1, 9, len(df_plot)) 
                    
                    Y_POS_FIXED = {1: 3, 2: 2, 3: 1}
                    fase_points = defaultdict(list)
                    
                    km_central = 0.0
                    imagem_torre_central = None 
                    current_code = ""

                    for index, row in df_plot.iterrows():
                        x = row["x_pos"]
                        raw_seq_or_code = str(row[fase_seq_col]).strip().upper()
                        
                        seq_fase_real = raw_seq_or_code 
                        tower_label = str(row[desc_col]).strip()
                        caminho_imagem = None 
                        is_brasnorte = concessao_escolhida == "BRASNORTE"
                        
                        if is_brasnorte and raw_seq_or_code in torres_jbju_map:
                            figura_ref, seq_fase_real_map, caminho_imagem_map = torres_jbju_map.get(raw_seq_or_code, (str(row[desc_col]).strip(), raw_seq_or_code, None))
                            seq_fase_real = seq_fase_real_map
                            caminho_imagem = caminho_imagem_map
                            tower_label = f"Torre: {figura_ref}\n({raw_seq_or_code})"

                        if index == idx_central:
                            km_central = row[km_col]
                            x_central = x
                            imagem_torre_central = caminho_imagem 
                            current_code = raw_seq_or_code 
                            if is_brasnorte and current_code in torres_jbju_map:
                                _, _, _ = torres_jbju_map[current_code]
                                current_code = raw_seq_or_code

                        if len(seq_fase_real) == 3:
                            fases_na_torre = {
                                seq_fase_real[0]: Y_POS_FIXED[1], 
                                seq_fase_real[1]: Y_POS_FIXED[2], 
                                seq_fase_real[2]: Y_POS_FIXED[3]  
                            }
                            for fase_letra, y_pos in fases_na_torre.items():
                                fase_points[fase_letra].append((x, y_pos))
                        

                    # --- Plotagem (Gráfico) ---
                    col_fig, col_tabela, col_imagem = st.columns([2, 1, 1])
                    with col_fig:
                        fig, ax = plt.subplots(figsize=(12, 7))
                        ax.set_xlim(0, 10)
                        ax.set_ylim(0, 4)
                        ax.axis("off") 
                        
                        y_start_torre = 0.8
                        y_end_torre = 3.2
                        
                        FASE_COLORS = {"A": "orange", "B": "green", "C": "purple"}
                        
                        # 1. Desenha as Linhas de Fase (Transposição)
                        for fase_letra, points in fase_points.items():
                            if points:
                                x_coords = [p[0] for p in points]
                                y_coords = [p[1] for p in points]
                                
                                color = FASE_COLORS.get(fase_letra, "gray")
                                linewidth = 3 if fase_letra == fase_escolhida else 1.5
                                linestyle = '-' if fase_letra == fase_escolhida else '--'
                                
                                ax.plot(x_coords, y_coords, color=color, linewidth=linewidth, linestyle=linestyle, alpha=0.7, zorder=1)
                                
                                if len(x_coords) > 0:
                                    ax.text(x_coords[-1] + 0.1, y_coords[-1], f"Fase {fase_letra}", va="center", fontsize=10, color=color)

                        # 2. Desenha as Torres e Rótulos (Mantido)
                        for index, row in df_plot.iterrows():
                            x = row["x_pos"]
                            is_central = index == idx_central
                            
                            line_color = "red" if is_central else "gray"
                            line_style = "-" if is_central else "--"
                            line_width = 3 if is_central else 1.5

                            ax.vlines(x, y_start_torre, y_end_torre, 
                                      colors=line_color, linestyles=line_style, linewidth=line_width, zorder=3)
                            
                            km_text = f"{row[km_col]:.2f} km"
                            
                            current_code_plot = str(row[fase_seq_col]).strip().upper()
                            tower_label = str(row[desc_col]).strip()
                            seq_to_display = current_code_plot
                            
                            if is_brasnorte and current_code_plot in torres_jbju_map:
                                figura_ref, seq_fase_real, _ = torres_jbju_map[current_code_plot]
                                tower_label = f"Torre: {figura_ref}\n({current_code_plot})" 
                                seq_to_display = seq_fase_real

                            ax.text(x, 0.7, f"{tower_label}\n{km_text}", ha="center", fontsize=9, color=line_color if is_central else "black")
                            
                            ax.text(x, y_end_torre + 0.1, f"Seq: {seq_to_display}", ha="center", fontsize=9, 
                                    bbox=dict(facecolor='white', alpha=0.8, edgecolor=line_color if is_central else 'gray', boxstyle='round,pad=0.3'), zorder=4)


                        # 3. Desenha o KM de Busca (Mantido)
                        x_busca = x_central
                        # (Lógica de interpolação mantida)
                        if valor_busca != km_central:
                            torre_ant = df_lt[(df_lt[km_col] < valor_busca)].iloc[-1] if not df_lt[df_lt[km_col] < valor_busca].empty else None
                            torre_prox = df_lt[(df_lt[km_col] >= valor_busca)].iloc[0] if not df_lt[df_lt[km_col] >= valor_busca].empty else None
                            
                            if torre_ant is not None and torre_prox is not None:
                                km_ant = torre_ant[km_col]
                                km_prox = torre_prox[km_col]
                                
                                x_ant_idx = df_plot.index[df_plot[km_col] == km_ant].tolist()
                                x_prox_idx = df_plot.index[df_plot[km_col] == km_prox].tolist()

                                if x_ant_idx and x_prox_idx and km_prox > km_ant:
                                    x_ant = df_plot.loc[x_ant_idx[0], "x_pos"]
                                    x_prox = df_plot.loc[x_prox_idx[0], "x_pos"]
                                    
                                    distancia_total = km_prox - km_ant
                                    distancia_relativa = valor_busca - km_ant
                                    proporcao = distancia_relativa / distancia_total
                                    x_busca = x_ant + proporcao * (x_prox - x_ant)

                        ax.vlines(x_busca, y_start_torre, y_end_torre, colors="blue", linestyles="dotted", linewidth=2, zorder=5)
                        ax.text(x_busca, 0.4, f"KM de Busca: {valor_busca:.2f}", ha="center", color="blue", fontsize=10, 
                                bbox=dict(facecolor='lightblue', alpha=0.7, edgecolor='blue', boxstyle='round,pad=0.3'), zorder=6)
                        
                        # Destaque do PONTO do KM de busca na fase afetada (Mantido)
                        target_fase_points = fase_points.get(fase_escolhida)
                        
                        if target_fase_points:
                            x_coords = [p[0] for p in target_fase_points]
                            y_coords = [p[1] for p in target_fase_points]
                            
                            for i in range(len(x_coords) - 1):
                                if x_coords[i] <= x_busca <= x_coords[i+1]:
                                    x1, y1 = x_coords[i], y_coords[i]
                                    x2, y2 = x_coords[i+1], y_coords[i+1]
                                    
                                    if x2 - x1 != 0:
                                        y_busca = y1 + (y2 - y1) * (x_busca - x1) / (x2 - x1)
                                        ax.plot(x_busca, y_busca, 'o', markersize=10, color='red', markeredgecolor='black', zorder=10)
                                        break
                        
                        st.pyplot(fig) 
                    
                    # --- Exibição da Imagem da Torre Central (MANTIDO) ---
                    with col_imagem:
                        st.markdown("### 🖼️ Figura da Torre")
                        if imagem_torre_central and imagem_torre_central.strip():
                            caminho_completo = imagem_torre_central
                            
                            try:
                                if os.path.exists(caminho_completo):
                                    st.image(caminho_completo, caption=f"Torre {current_code}", use_container_width=True)
                                else:
                                    st.warning(f"❌ Arquivo não encontrado: `{caminho_completo}`. Verifique o caminho na Coluna E e se o arquivo existe.")
                            except Exception as img_e:
                                st.error(f"❌ Erro ao carregar a imagem: {img_e}")
                        else:
                            st.info("Caminho da imagem não especificado na Coluna E da planilha 'Torres JBJU' para esta torre.")

                    # --- Tabela da Janela de Inspeção (MANTIDO) ---
                    with col_tabela:
                        if comprimento is not None and comprimento > 0:
                            perc = 0.02 if metodo in ["SIGRA 2 Terminais", "Sequência Negativa"] else 0.05
                            km_ini = max(0, valor_busca - comprimento * perc)
                            km_fim = valor_busca + comprimento * perc
                            
                            torres_na_janela_df = df_lt[
                                (df_lt[km_col] >= km_ini) &
                                (df_lt[km_col] <= km_fim)
                            ].copy()
                            
                            janela_df = pd.DataFrame({
                                "Janela de Inspeção": ["KM Inicial", "KM de Busca", "KM Final", "Porcentagem", "Torres na Janela"],
                                "Valor": [f"{km_ini:.2f} km", f"{valor_busca:.2f} km", f"{km_fim:.2f} km", f"{perc*100:.0f}%", f"{len(torres_na_janela_df)}"]
                            })
                            st.markdown("### 📋 Janela de Inspeção")
                            st.dataframe(janela_df, hide_index=True, use_container_width=True)
                            
                else:
                    st.warning("⚠️ Nenhuma torre encontrada para esse KM ou KM fora do limite da LT.")

            elif plotar_clicado and valor_busca == 0:
                st.warning("⚠️ O KM de Busca não pode ser zero. Insira um valor para plotar.")

        else:
            st.info("👆 Escolha uma Concessão e uma LT para continuar.")

    except Exception as e:
        # st.exception(e) 
        st.error(f"❌ Ocorreu um erro ao processar o arquivo. Verifique se as abas e colunas estão corretas. Detalhe: {e}")

else:
    st.info("👆 Faça o upload de um arquivo Excel para começar.")