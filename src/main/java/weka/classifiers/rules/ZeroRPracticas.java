/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    ZeroR.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

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
 * <!-- globalinfo-start --> Class for building and using a 0-R classifier.
 * Predicts the mean (for a numeric class) or the mode (for a nominal class).
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 * 
 * <!-- options-end -->
 * 
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 10153 $
 */
public class ZeroRPracticas extends AbstractClassifier implements
		WeightedInstancesHandler, Sourcable, OptionHandler {

	/** for serialization */
	static final long serialVersionUID = 48055541465867954L;

	/** The class value 0R predicts. */
	private double m_ClassValue;

	/** The number of instances in each class (null if class numeric). */
	private double[] m_Counts;

	/** The class attribute. */
	private Attribute m_Class;

	/** Prototipos de cada clase */
	private List<ArrayList<Instance>> prototipos;

	/** Numero de prototipos (parametro) */
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
	 * entrenamiento. El numero de prototipos que se creen se introduce como
	 * parametro.
	 * 
	 * @param numPrototipos
	 * @return
	 */
	/**
	 * Generates the classifier.
	 * 
	 * @param instances
	 *            set of instances serving as training data
	 * @throws Exception
	 *             if the classifier has not been generated successfully
	 */
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		/*
		 * // can classifier handle the data?
		 * getCapabilities().testWithFail(instances);
		 * 
		 * // remove instances with missing class instances = new
		 * Instances(instances); instances.deleteWithMissingClass();
		 * 
		 * double sumOfWeights = 0;
		 * 
		 * m_Class = instances.classAttribute(); m_ClassValue = 0; switch
		 * (instances.classAttribute().type()) { case Attribute.NUMERIC:
		 * m_Counts = null; break; case Attribute.NOMINAL: m_Counts = new
		 * double[instances.numClasses()]; for (int i = 0; i < m_Counts.length;
		 * i++) { m_Counts[i] = 1; } sumOfWeights = instances.numClasses();
		 * break; } Enumeration<Instance> enu = instances.enumerateInstances();
		 * while (enu.hasMoreElements()) { Instance instance =
		 * enu.nextElement(); if (!instance.classIsMissing()) { if
		 * (instances.classAttribute().isNominal()) { m_Counts[(int)
		 * instance.classValue()] += instance.weight(); } else { m_ClassValue +=
		 * instance.weight() * instance.classValue(); } sumOfWeights +=
		 * instance.weight(); } } if (instances.classAttribute().isNumeric()) {
		 * if (Utils.gr(sumOfWeights, 0)) { m_ClassValue /= sumOfWeights; } }
		 * else { m_ClassValue = Utils.maxIndex(m_Counts);
		 * Utils.normalize(m_Counts, sumOfWeights); }
		 */

		int filas = instances.numClasses();
		int columnas = numPrototipos;

		ArrayList<Integer> numProtSeleccionados = new ArrayList<Integer>(filas);

		for (int i = 0; i < filas; ++i) {
			numProtSeleccionados.add(0);
		}

		prototipos = new ArrayList<ArrayList<Instance>>(filas);

		for (int i = 0; i < filas; ++i) {
			prototipos.add(new ArrayList<Instance>(columnas));
		}

		Enumeration<Instance> enu = instances.enumerateInstances();
		Instance instance;
		Attribute tipoClass;
		// TODO
		/* No hace falta recorrer todo el set */
		for (int i = 0; enu.hasMoreElements(); ++i) {
			instance = enu.nextElement();
			tipoClass = instance.classAttribute();

			/*
			 * if (numProtSeleccionados.get(tipoClass) < numPrototipos) {
			 * prototipos.get(tipoClass).add(instance); }
			 */
		}

	}

	public void entrenaPrototipos(Instances instances) {
		Enumeration<Instance> enu = instances.enumerateInstances();
		Instance instance;
		int tipoClass;

		for (int i = 0; enu.hasMoreElements(); ++i) {
			instance = enu.nextElement();
			tipoClass = instance.classAttribute().type();

			// Calcular el prototipo mas cercano
			double distMin = Double.MIN_VALUE;
			double distAct;
			Instance masCercano = null;

			for (int j = 0; j < numPrototipos; ++j) {
				distAct = distancia(prototipos.get(tipoClass).get(j), instance);

				if (distAct < distMin) {
					distMin = distAct;
					masCercano = prototipos.get(tipoClass).get(j);
				}
			}
			// TODO
			// Comprobar si cambian los valores originales

			for (int k = 0; k < masCercano.numAttributes(); ++k) {

			/*	switch (masCercano.attribute(k).type()) {
				case Attribute.NUMERIC:
					m_Counts = null;
					break;
				case Attribute.NOMINAL:
					m_Counts = new double[instances.numClasses()];
					for (int i = 0; i < m_Counts.length; i++) {
						m_Counts[i] = 1;
					}
					sumOfWeights = instances.numClasses();
					break;
				}*/
			}
		}
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

		for (int i = 0; i < array1.length; ++i) {
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

		return m_ClassValue;
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

		if (m_Counts == null) {
			double[] result = new double[1];
			result[0] = m_ClassValue;
			return result;
		} else {
			return m_Counts.clone();
		}
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
			// TODO Auto-generated catch block
			e.printStackTrace();

		}
	}

	@Override
	public String[] getOptions() {
		Vector<String> options = new Vector<String>();
		options.add("-I");
		options.add("" + getNumPrototipos());

		Collections.addAll(options, super.getOptions());
		// TODO
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
		runClassifier(new ZeroRPracticas(), argv);
	}
}
