# Predicción de fuga de clientes (Churn) en una telecom

Modelo de clasificación para anticipar qué clientes de **Interconnect**, un operador de telecomunicaciones, se darán de baja, de modo que marketing pueda ofrecerles promociones a tiempo.

**Resultado final (conjunto de prueba):** AUC-ROC **0,843** · Recall **0,872** · Precision 0,450

> Proyecto final de bootcamp de Data Science (aprobado). Notebook: [`notebooks/proyecto_churn_interconnect.ipynb`](notebooks/proyecto_churn_interconnect.ipynb)

## Problema de negocio

Perder un cliente significa perder ingresos. Un falso negativo (un cliente que se va y el modelo dijo que se quedaba) es el error más caro, porque no se le ofreció ninguna promoción. Por eso las métricas son:

- **AUC-ROC** (principal): capacidad del modelo de ordenar a los clientes por riesgo.
- **Recall** (secundaria): qué porcentaje de los clientes que se van logra detectar el modelo.

## Datos

Cuatro tablas unidas por `customerID` (7.043 clientes): contrato, datos personales, servicios de internet y servicios de teléfono. La tasa de abandono es 26,5 %.

> Los datos pertenecen al curso y no se incluyen en este repositorio. Para ejecutar el notebook, coloca los CSV (`contract.csv`, `personal.csv`, `internet.csv`, `phone.csv`) en la carpeta `data/` y ajusta las rutas de la celda de carga.

## Metodología

1. **EDA:** tasas de abandono por categoría, relaciones entre variables y trazabilidad de las bajas en el tiempo.
2. **Preprocesamiento:** unión con *left join*, nulos de servicios no contratados como categoría explícita, *one-hot encoding* dentro de un `Pipeline`.
3. **Baseline:** Regresión Logística y Random Forest sin balanceo.
4. **Optimización:** LightGBM con `class_weight="balanced"`, búsqueda aleatoria de hiperparámetros con validación cruzada estratificada y umbral de decisión elegido solo con datos de entrenamiento.

## Hallazgo clave: detección de fuga de datos

Una primera versión con la duración del contrato (`DuracionDias`) alcanzó un AUC de 0,906. Ese valor estaba **inflado**: para los clientes que se fueron, la duración se calcula hasta su fecha de baja, información que no existe antes de que el cliente se vaya.

| Variables excluidas | AUC test | Recall |
|---|---|---|
| Ninguna | 0,906 | 0,837 |
| `DuracionDias` (**modelo final**) | **0,843** | **0,872** |
| `DuracionDias` + `TotalCharges` | 0,825 | 0,874 |

Se eligió la versión sin fuga, que es la cifra que se puede esperar en producción.

## Principales resultados

- **Perfil de mayor riesgo:** contrato mes a mes (42,7 % de abandono frente a 2,8 % en contratos de dos años), fibra óptica, pago con cheque electrónico, sin soporte técnico ni seguridad en línea, y adultos mayores.
- **Impacto:** con Recall 0,872 el modelo detecta unos 87 de cada 100 clientes que se irían, marcando a cerca de la mitad de la base, con una tasa de abandono ~1,7 veces la general.

## Limitaciones

- `TotalCharges` aún arrastra parte de la información de antigüedad.
- Los datos son una foto de febrero de 2020 y no se midió el efecto causal de las promociones; se recomienda una prueba A/B.

## Cómo ejecutarlo

```bash
git clone https://github.com/<tu-usuario>/proyecto-churn-interconnect.git
cd proyecto-churn-interconnect
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/proyecto_churn_interconnect.ipynb
```

## Tecnologías

Python · pandas · NumPy · scikit-learn · LightGBM · matplotlib · seaborn · Jupyter

## Autor

**<constanza silva>** · [LinkedIn](https://www.linkedin.com/in/constanza-silva-ba%C3%A9z-758074178/)
