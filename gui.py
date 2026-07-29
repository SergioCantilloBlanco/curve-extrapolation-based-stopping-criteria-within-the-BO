import gradio as gr
import pandas as pd

DATASETS = {
    "Ventas 2025": "ruta/a/tus_datos/ventas.csv",
    "Clientes": "https://githubusercontent.com",
    "Resultados Test": "ruta/a/tus_datos/test.xlsx"
}

def cargar_y_procesar_dataset(nombre_dataset):
    try:
        ruta = DATASETS[nombre_dataset]
        
        if ruta.endswith('.csv'):
            df = pd.read_csv(ruta)
        elif ruta.endswith('.xlsx') or ruta.endswith('.xls'):
            df = pd.read_excel(ruta)
        else:
            return f"Error: Formato de archivo no soportado para {nombre_dataset}.", None
        
        # Aquí puedes añadir tu pipeline de Machine Learning (limpieza, split, etc.)
        resumen = f"✅ Dataset '{nombre_dataset}' cargado con éxito.\n📊 Filas: {df.shape[0]} | Columnas: {df.shape[1]}"
        
        # Retorna el texto informativo y las primeras filas para la interfaz
        return resumen, df.head(10)
        
    except Exception as e:
        return f"❌ Error al cargar el dataset: {str(e)}", None

# 3. Diseña la interfaz gráfica con Gradio
with gr.Blocks(title="ML Data Loader") as demo:
    gr.Markdown("#Selector de Datasets")
    gr.Markdown("Selecciona un conjunto de datos y haz clic para integrarlo instantáneamente en el proyecto.")
    
    with gr.Row():
        # Menú desplegable con las opciones de tus datasets
        dataset_selector = gr.Dropdown(
            choices=list(DATASETS.keys()), 
            label="Elige un Dataset", 
            value=list(DATASETS.keys())[0]
        )
        # Botón de acción principal
        btn_cargar = gr.Button("Cargar Dataset", variant="primary")
    
    # Componentes de salida
    texto_estado = gr.Textbox(label="Estado del Sistema", placeholder="Esperando selección...")
    vista_datos = gr.Dataframe(label="Vista previa de los datos (Top 10 filas)")
    
    # 4. Conecta el clic del botón con la función lógica
    btn_cargar.click(
        fn=cargar_y_procesar_dataset, 
        inputs=dataset_selector, 
        outputs=[texto_estado, vista_datos]
    )

# 5. Lanza la aplicación web local
if __name__ == "__main__":
    demo.launch()