# Antecedentes / estado del arte

El estudio de las brechas laborales por **género** y por **edad** en Argentina
tiene una tradición consolidada sobre la misma fuente que usamos aquí (la EPH del
INDEC), lo que respalda tanto el dataset como la metodología elegida.

**Brecha de género e ingresos (Oaxaca–Blinder sobre EPH).** La descomposición de
Oaxaca–Blinder aplicada a la EPH es el enfoque estándar para separar la parte de
la brecha salarial explicada por características (educación, experiencia, horas,
sector) de la parte no explicada (asociada a discriminación). El CEDLAS (UNLP) y
la Dirección de Economía, Igualdad y Género del Ministerio de Economía han
documentado que la brecha persiste incluso al controlar por capital humano y
segmentación ocupacional. Nuestro trabajo replica este enfoque (Sección de
Oaxaca) y aporta el cruce **edad × género** y la **dimensión temporal 2016–2024**.

**Discriminación por edad / edadismo.** Arese (2020) analiza el "edadismo" laboral
y previsional desde el derecho: lo caracteriza como una discriminación
generalmente invisible, recuerda antecedentes como la *Age Discrimination in
Employment Act* de EE.UU. (que protege a los mayores de 40) y subraya la
dimensión **previsional** del problema — el mismo mecanismo institucional que
discutimos (aceptar un empleo subpago cerca de la jubilación deteriora la base de
cálculo del haber). Oddone (CEIL-CONICET) aborda empíricamente a los trabajadores
de mayor edad y su "desprendimiento laboral" hacia el cuentapropismo, en línea con
nuestro mecanismo de **fuga al cuentapropismo**.

**Evidencia social.** Encuestas recientes (p. ej. Bumeran, 2024) reportan que ~61%
de los trabajadores sufrió o presenció discriminación por edad, cifra que sube a
~86% entre los mayores de 40. Es evidencia de percepción que contextualiza, pero
no reemplaza, el análisis cuantitativo.

**Nuestro aporte frente a la literatura.** (1) Integramos en un mismo trabajo la
brecha de **ingreso** y la de **empleo/exclusión**; (2) explotamos la **serie
2016–2024** para ver la evolución de las brechas; (3) combinamos econometría
clásica (Oaxaca) con **machine learning interpretable** (partial dependence,
SHAP) y una **auditoría de fairness**; y (4) discutimos el **mecanismo
previsional** como factor de la no-reinserción de los seniors.

## Referencias

- Arese, C. (2020). *El "edadismo" laboral y previsional.* Revista Derecho de las
  Minorías, 3, 138–. Universidad Católica de Córdoba.
  DOI: 10.22529/rdm.2020(3)05. **[leída]**
- Oddone, M. J. *Los trabajadores de mayor edad: empleo y desprendimiento
  laboral.* CEIL-CONICET. **[verificar año y contenido — PDF escaneado]**
- CEDLAS (UNLP). *Brechas de Género: Una Exploración Más Allá de la Media.*
  Documento de Trabajo CEDLAS. **[verificar autores/nº]**
- Ministerio de Economía (Dir. de Economía, Igualdad y Género). *Brecha salarial
  de género en la estructura productiva argentina.* Documento de Trabajo 2.
  **[verificar]**
- Universidad Torcuato Di Tella. *¿En qué medida persiste la desigualdad salarial
  en Argentina? La brecha de género en el mercado laboral argentino.*
  Repositorio UTDT. **[verificar]**
- *La brecha salarial por género en Argentina: un análisis acerca de la
  segmentación laboral.* Redalyc (journal 703). **[verificar]**
- Bumeran (2024). *Encuesta sobre discriminación laboral por edad.* (fuente
  periodística: Infobae, 2/6/2024). **[evidencia de percepción]**
