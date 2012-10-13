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
 * \brief Implementation of max plane class
 * \author Akhmed Umyarov
 * \date 2007
 */

#include "cvmaxplane.h"
#include <iostream>
#include <sstream>

using namespace std;

// Constructors/Destructors
//  

/*!
 * Constructor creates a new max plane with specified name.
 * Since max plane is just a data abstraction, feature map and 
 * neuron window size are irrelevant and are not used in the constructor
 * \param id name of the plane
 */
CvMaxPlane::CvMaxPlane (std::string id)
	: CvGenericPlane(id, cvSize(1,1), cvSize(1,1) ) 
{
	m_weight.clear();
}

CvMaxPlane::~CvMaxPlane ( ) 
{
}

//  
// Methods
//  

/*! The method forward-propagates data from neuron parents to neuron's 
 * own feature map.
 * Each subsampling neuron knows his parents, thus there are no input parameters.
 * \return Pointer to plane's featuremap
 */
CvMat * CvMaxPlane::fprop ( )
{
	assert( m_connected );
	if (!m_connected)
	{
#ifdef DEBUG
		cout << "CvMaxPlane::fprop(): Not connected!" << endl;
#endif
		return NULL;
	}

	// Get the values at parent planes
	int no_parents = m_pplane.size();
	m_parentval.resize( no_parents );
	for (int i = 0; i < no_parents; i++)
	{
		m_parentval[i] = cvmGet( m_pfmap[i], 0, 0 );
	}

	// Now find the maximum of m_parentval
	vector<double>::iterator itr = max_element(m_parentval.begin(),m_parentval.end());

	// The index of maximum is our network's prediction!
	int pos = distance(m_parentval.begin(), itr);

	cvmSet(m_fmap,0,0,(double) pos );

	return m_fmap;
}


/*! The method produces an XML representation of the complete information about 
 * the plane including information about weights of neuron and connection to
 * parents.
 * The representation does NOT include current state of the plane
 * such as its feature map.
 * The method is mainly used by the CvConvNet object.
 * \return string containing XML description of the plane
 */
string CvMaxPlane::toString ( ) 
{
	ostringstream xml;
 	xml << "\t<plane id=\"" << m_id << "\" type=\"max\">" << endl;
	
	for (int i=0; i <m_pplane.size(); i++)
	{
		xml << "\t\t<connection to=\"" << m_pplane[i]->getid() << "\"> " << endl;
		xml << "</connection>" << endl;
	}
	xml << "\t</plane>" << endl;
	
	return xml.str();
}

/*! The method explicitly sets the weights of the neuron
 */
int CvMaxPlane::setweight(std::vector<double> &weights)
{	
	// Just dummy function
	return 1;
}
