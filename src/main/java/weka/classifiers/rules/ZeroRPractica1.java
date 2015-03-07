package weka.classifiers.rules;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Sourcable;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

/**
 * <!-- globalinfo-start --> Clasificador basado en prototipos.
 * <p/>
 * Funcional unicamente cuando los atributos de las instancias(a excepción del
 * atributo de clase) son numéricos.
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -I
 * 
 *  Indica el número de prototipos (de cada clase) que usaremos en el clasificador.
 * </pre>
 * 
 * <!-- options-end -->
 * 
 * @author Adrián Arroyo Pérez (aap0065@alu.ubu.es) Alejandro González Rogel
 *         (agr0095@alu.ubu.es)
 * 
 * @version 1.0
 */
public class ZeroRPractica1 extends AbstractClassifier implements
		WeightedInstancesHandler, Sourcable, OptionHandler {

	/** for serialization */
	static final long serialVersionUID = 48055541465867954L;

	/** The class value 0R predicts. */
	private double m_ClassValue;

	/** The number of instances in each class (null if class numeric). */
	private double[] m_Counts;

	/** The class attribute. */
	private Attribute m_Class;

	/**
	 * Matriz para almacenar todos nuestros prototipos. El número de filas
	 * corresponde al número de clases existentes en el dataset. El número de
	 * columnas corresponde al número de prototipos para cada clase.
	 * */
	private List<ArrayList<Instance>> prototipos;

	/** Numero de prototipos de cada posible clase existente en el dataset */
	private int numPrototipos;

	/**
	 * Returns a string describing classifier
	 * 
	 * @return a description suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Class for building and using a 0-R classifier. Predicts the mean "
				+ "(for a numeric class) or the mode (for a nominal class).";
	}

	/**
	 * Returns default capabilities of the classifier.
	 * 
	 * @return the capabilities of this classifier
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.DATE_ATTRIBUTES);
		result.enable(Capability.STRING_ATTRIBUTES);
		result.enable(Capability.RELATIONAL_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NUMERIC_CLASS);
		result.enable(Capability.DATE_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		// instances
		result.setMinimumNumberInstances(0);

		return result;
	}

	/**
	 * Crea una serie de prototipos de cada clase a partir de los datos de
	 * entrenamiento. Además, una vez seleccionados los prototipos, se entrenan.
	 * 
	 * El numero de prototipos creados se ha introducido por parámetro en la
	 * invocación del método main y está almacenado en @numPrototipos.
	 * 
	 * @param instances
	 *            set of instances serving as training data
	 * @throws Exception
	 *             if the classifier has not been generated successfully
	 */
	@Override
	public void buildClassifier(Instances instances) throws Exception {

		// Inicializamos la matriz que contendrá los prototipos.
		prototipos = new ArrayList<ArrayList<Instance>>(instances.numClasses());
		for (int i = 0; i < instances.numClasses(); ++i) {
			prototipos.add(new ArrayList<Instance>());
		}

		Enumeration<Instance> enu = instances.enumerateInstances();
		Instance instance;

		// TODO
		// Los prototipos de consiguen de manera lineal, no aleatoria

		// Añadimos un prototipo de la clase de la instancia (si podemos)
		for (; enu.hasMoreElements() && !prototiposCompletos();) {
			instance = enu.nextElement();

			if (prototipos.get((int) instance.classValue()).size() < numPrototipos) {
				prototipos.get((int) instance.classValue()).add(instance);
			}

		}

		// Si no hemos podido generar correctamente el clasificador, lanzamos un
		// error
		if (!enu.hasMoreElements() && prototiposCompletos())
			throw new Exception("Excepción en la creación del clasificador");

		entrenaPrototipos(instances);

	}

	/**
	 * Comprueba si hemos almacenado todos los prototipos que se nos pedían.
	 * 
	 * @return true si tenemos almacenados todos los prototipos. false si no
	 *         tenemos todos los prototipos
	 */
	private boolean prototiposCompletos() {
		for (int i = 0; i < prototipos.size(); i++) {
			if (prototipos.get(i).size() != numPrototipos)
				return false;
		}
		return true;
	}

	/**
	 * Entrena nuestro conjunto de prototipos para que se distribuyan por el
	 * espacio muestral que corresponde a la clase a la que pertenecen.
	 * 
	 * @param instances
	 *            El conjunto total de datos con los que contamos.
	 */
	public void entrenaPrototipos(Instances instances) {

		Enumeration<Instance> enu = instances.enumerateInstances();
		Instance instance;
		Instance masCercano; // Almacena el prototipo más cercano a la instancia
								// concreta.

		// Por cada instancia
		for (; enu.hasMoreElements();) {

			instance = enu.nextElement();
			// Calculamos el prototipo más semejante a la instancia
			masCercano = prototipoMasCercano(instance);

			// Actualizamos el valor de nuestro prototipo
			actualizarPrototipo(instance, masCercano);

		}
	}

	/**
	 * Actualiza el valor del prototipo pasado por parámetro tomando como
	 * referencia una instancia también pasada por parámetro
	 *
	 * @param instance
	 *            Instancia con la que actualizaremos el prototipo prototipo
	 *            Prototipo a actualizar.
	 *
	 */

	public void actualizarPrototipo(Instance instance, Instance prototipo) {

		double[] valores1 = prototipo.toDoubleArray();
		double[] valores2 = instance.toDoubleArray();

		// Actualizamos el prototipo
		for (int k = 0; k < instance.numAttributes() - 1; ++k) {
			prototipo.setValue(k, (valores1[k] + valores2[k]) / 2);
		}

	}

	/**
	 * Devuelve el prototipo más parecido a la instancia pasada por parámetro.
	 * Conocemos la clase de la instancia pasada por parámetro.
	 * 
	 * @param instance
	 *            Instancia de la que queremos calcular el prototipo más cercano
	 * @return prototipo más cercano.
	 */
	public Instance prototipoMasCercano(Instance instance) {

		double distMin = Double.MAX_VALUE;
		double distAct = Double.MAX_VALUE;
		Instance masCercano = null;

		// Calculamos el prototipo mas cercano

		for (int k = 0; k < numPrototipos; ++k) {
			// TODO
			// No esta implementado para atributos nominales
			distAct = distancia(prototipos.get((int) instance.classValue())
					.get(k), instance);

			if (distAct < distMin) {
				distMin = distAct;
				masCercano = prototipos.get((int) instance.classValue()).get(k);
			}
		}

		return masCercano;
	}

	/**
	 * Calcula la distancia euclidea entre los valores de los atributos de dos
	 * instancias que le vienen como parametro.
	 * 
	 * @param instance1
	 * @param instance2
	 * @return
	 */
	public double distancia(Instance instance1, Instance instance2) {
		double[] array1 = instance1.toDoubleArray();
		double[] array2 = instance2.toDoubleArray();
		double distance = 0;

		for (int i = 0; i < array1.length - 1; ++i) { // -1 para evitar comparar
														// la distancia que
														// separa el atributo
														// clase
			distance += Math.pow(Math.abs(array2[i] - array1[i]), 2);
		}

		return distance;
	}

	/**
	 * Classifies a given instance.
	 * 
	 * @param instance
	 *            the instance to be classified
	 * @return index of the predicted class
	 */
	@Override
	public double classifyInstance(Instance instance) {

		double distMin = Double.MAX_VALUE;
		double distAct;
		Instance masCercano = null;

		for (int i = 0; i < prototipos.size(); ++i) {

			for (int j = 0; j < numPrototipos; ++j) {
				// TODO
				// No esta implementado para atributos nominales
				distAct = distancia(prototipos.get(i).get(j), instance);

				if (distAct < distMin) {
					distMin = distAct;
					masCercano = prototipos.get(i).get(j);
				}
			}
		}

		return masCercano.classValue();

	}

	/**
	 * Calculates the class membership probabilities for the given test
	 * instance.
	 * 
	 * @param instance
	 *            the instance to be classified
	 * @return predicted class probability distribution
	 * @throws Exception
	 *             if class is numeric
	 */
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {

		//TODO
		//ARROYO, dijiste que tenías algo apuntado sobre este método...pues mira a ver que es.
		double valorClasf = classifyInstance(instance);

		double[] probabilidades = new double[prototipos.size()];

		for (int i = 0; i < probabilidades.length; ++i) {
			if (valorClasf == i) {
				probabilidades[i] = 1;
			} else {
				probabilidades[i] = 0;
			}
		}

		return probabilidades;
	}

	/**
	 * Returns a string that describes the classifier as source. The classifier
	 * will be contained in a class with the given name (there may be auxiliary
	 * classes), and will contain a method with the signature:
	 * 
	 * <pre>
	 * <code>
	 * public static double classify(Object[] i);
	 * </code>
	 * </pre>
	 * 
	 * where the array <code>i</code> contains elements that are either Double,
	 * String, with missing values represented as null. The generated code is
	 * public domain and comes with no warranty.
	 * 
	 * @param className
	 *            the name that should be given to the source class.
	 * @return the object source described by a string
	 * @throws Exception
	 *             if the souce can't be computed
	 */
	@Override
	public String toSource(String className) throws Exception {
		StringBuffer result;

		result = new StringBuffer();

		result.append("class " + className + " {\n");
		result.append("  public static double classify(Object[] i) {\n");
		if (m_Counts != null) {
			result.append("    // always predicts label '"
					+ m_Class.value((int) m_ClassValue) + "'\n");
		}
		result.append("    return " + m_ClassValue + ";\n");
		result.append("  }\n");
		result.append("}\n");

		return result.toString();
	}

	/**
	 * Returns a description of the classifier.
	 * 
	 * @return a description of the classifier as a string.
	 */
	@Override
	public String toString() {

		if (m_Class == null) {
			return "ZeroR: No model built yet.";
		}
		if (m_Counts == null) {
			return "ZeroR predicts class value: " + m_ClassValue;
		} else {
			return "ZeroR predicts class value: "
					+ m_Class.value((int) m_ClassValue);
		}
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	@Override
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 10153 $");
	}

	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>(1);

		newVector.addElement(new Option("\t Number of prototypes \n"
				+ "\t(use when prototypes > 1)", "I", 0, "-I"));
		return newVector.elements();
	}

	@Override
	public void setOptions(String[] options) {
		String prototipos;
		try {
			prototipos = Utils.getOption('I', options);
			if (prototipos.length() != 0) {
				setNumPrototipos(Integer.parseInt(prototipos));
			} else {
				setNumPrototipos(1);
			}
		} catch (Exception e) {
			e.printStackTrace();

		}
	}

	@Override
	public String[] getOptions() {
		Vector<String> options = new Vector<String>();
		options.add("-I");
		options.add("" + getNumPrototipos());

		Collections.addAll(options, super.getOptions());
		return options.toArray(new String[0]);
	}

	public int getNumPrototipos() {
		return numPrototipos;
	}

	public void setNumPrototipos(int numPrototipos) {
		this.numPrototipos = numPrototipos;
	}

	public String numPrototiposTipText() {
		return "Prototipos";
	}

	/**
	 * Main method for testing this class.
	 * 
	 * @param argv
	 *            the options
	 */
	public static void main(String[] argv) {
		runClassifier(new ZeroRPractica1(), argv);
	}
}
