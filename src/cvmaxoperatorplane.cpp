/*****************************************************************************
 IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING. By
downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install, copy or
use the software.

Contributors License Agreement

Copyright© 2007, Akhmed Umyarov. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.
- The name of Contributor may not be used to endorse or promote products derived
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
All information provided related to future Intel products and plans is
preliminary and subject to change at any time, without notice.
*****************************************************************************/

/*!\file
 * \brief Implementation of max operator plane class
 * \author Lloyd Henning
 * \date 2012
 */

#include "cvmaxoperatorplane.h"
#include "cvfastsigmoid.h"
#include <math.h>
#include <iostream>
#include <sstream>

using namespace std;

// Constructors/Destructors
//  

/*!
 * Constructor creates a new max operator plane with specified name,
 * specified size of feature map and specified size of "neuron window"
 * \param id name of the plane
 * \param fmapsz size of the featuremap for this plane
 * \param neurosz size of "neuron window" (for max operator planes, it is usually 2x2, 
 * that means that we have a neuron that is connected to 4 outputs of his 
 * parent neuron feature map)
 */
CvMaxOperatorPlane::CvMaxOperatorPlane (std::string id, CvSize fmapsz, CvSize neurosz)
	: CvGenericPlane(id, fmapsz, neurosz) 
{
	m_weight.resize( 2 );
	
}

CvMaxOperatorPlane::~CvMaxOperatorPlane ( ) 
{
}

//  
// Methods
//  

/*! The method forward-propagates data from neuron parents to neuron's 
 * own feature map.
 * Each max operator neuron knows his parents, thus there are no input parameters.
 * \return Pointer to plane's featuremap
 */
CvMat * CvMaxOperatorPlane::fprop()
{
    assert( m_connected );
    for (int row = 0; row < m_fmapsz.height / m_neurosz.height; row++)
    {
        for (int col = 0; col < m_fmapsz.width / m_neurosz.width; col++)
        {
            double max_so_far = -1000.0;
            // Probably only going to be one input feature map anyway
            for (int pfmap_index = 0; pfmap_index < m_pplane.size(); pfmap_index++)
            {
               CvMat *fmap = m_pfmap[pfmap_index]; 
               for (int filter_row = 0; filter_row < m_neurosz.height; filter_row++)
               {
                   for (int filter_col = 0; filter_col < m_neurosz.width; filter_col++)
                   {
                       double fmap_value = cvmGet(fmap, 
                                                  (row * m_neurosz.height)
                                                  + filter_row, 
                                                  (col * m_neurosz.width)
                                                  + filter_col);
                       if (fmap_value > max_so_far)
                       {
                           max_so_far = fmap_value;
                       } // if
                   } // for filter_col
               } // for filter_row
            } // for pfmap_index

            cvmSet(m_fmap, row, col, max_so_far);
        } // for col
    } // for row

    return m_fmap;
} // CvMaxOperatorPlane::fprop() 


/*! The method produces an XML representation of the complete information about 
 * the plane including information about weights of neuron and connection to
 * parents.
 * The representation does NOT include current state of the plane
 * such as its feature map.
 * The method is mainly used by the CvConvNet object.
 * \return string containing XML description of the plane
 */
string CvMaxOperatorPlane::toString ( ) 
{
	ostringstream xml;
 	xml << "\t<plane id=\"" << m_id << "\" type=\"max operator\" featuremapsize=\"" << m_fmapsz.width << "x" << m_fmapsz.height << "\" neuronsize=\"" << m_neurosz.width << "x" << m_neurosz.height << "\">" << endl;
	
	for (int i=0; i <m_pplane.size(); i++)
	{
		xml << "\t\t<bias> " << m_weight[0] << " </bias>" << endl;
		xml << "\t\t<connection to=\"" << m_pplane[i]->getid() << "\"> ";
		xml << m_weight[1] << " ";
		xml << "</connection>" << endl;
	}
	xml << "\t</plane>" << endl;
	
	return xml.str();
}

/*! The method explicitly sets the weights of the neuron
 */
int CvMaxOperatorPlane::setweight(std::vector<double> &weights)
{	
	// Check that the number of weights passed is sane
	if (weights.size() != 2 )
		return 0;

	return CvGenericPlane::setweight(weights);
}
