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
 * \brief Implementation
 *
 * Implementation
 * \author Akhmed Umyarov
 * \date 2007
 */

#include <cassert>
#include <iostream>
#include <sstream>
#include "cvconvnet.h"
#include "cvsourceplane.h"
#include "cvconvolutionplane.h"
#include "cvsubsamplingplane.h"
#include "cvgenericplane.h"
#include "cvrbfplane.h"
#include "cvconvnetparser.h"

using namespace std;
// Constructors/Destructors
//  

CvConvNet::CvConvNet ( )
{
	m_creator = "undefined";
	m_name = "untitled";
	m_info = "";
}

CvConvNet::~CvConvNet ( )
{ 
	// Clean up all planes
	for (int i = 0; i < m_plane.size(); i++)
	{
		assert( m_plane[i] != NULL );
		delete m_plane[i];
	}
	m_plane.clear();
}

//  
// Methods
//  

/*! The method propagates the input image through the whole network
 * updating planes (feature maps) of each individual neuron.
 * Each plane values can be accessed after fprop for further analysis.
 * \param  input pointer to input image in CvMat or IplImage format
 * \return (0,0) value of the last plane
 */
double CvConvNet::fprop (CvArr * input )
{
	if ( (input == NULL) || !(m_plane[0]->setfmap(input)) )
	{
		/*! \todo In case of wrong input, generate exception 
		 * instead of printing to cerr 
                 */
		cerr << "ERROR: Wrong input image" << endl;
		return -1.0;
	}
	
	// Iterate over all planes
	for (signed int i = 0; i < m_plane.size(); i++)
	{
		/*! \todo implement parallel fprop() invocation
		 * since independent planes can execute fprop 
		 * simultaneously!
		 */
		m_plane[i]->fprop();
	}

	return cvmGet(m_plane.back()->getfmap(),0,0);
}


/*! The method returns a pointer to matrix
 * of any individual feature map inside the network
 * The plane is specified by its text id (it is the same id
 * that is assigned to plane in XML file).
 * \param id String specifying the feature map to be accessed
 * \return pointer to CvMat structure of the specified plane
 */
const CvMat *CvConvNet::getplane( std::string id )
{
	map<string,int>::iterator itr = m_idmap.find(id); 
	assert( itr != m_idmap.end() );

	return m_plane[itr->second]->getfmap();
}

/*! Method produces an XML representation of the complete structure of
 * the convolutional network including information about connections 
 * between planes, weights for specific connections.
 * The representation does NOT include current state of the network
 * such as propagated values at each plane (feature map).
 * \return string containing XML description of the network
 */
std::string CvConvNet::toString()
{
	ostringstream xml;
	xml << *this;
	return xml.str();
}


/*! The method creates Convolutional Net from its string XML representation
 * \param xml XML-string representing the network
 * \return status of operation
 */
int CvConvNet::fromString ( std::string xml )
{
	return parse(xml, m_creator, m_name, m_info, m_plane, m_idmap);
}


ostream& operator<< (ostream& s, CvConvNet& n)
{
	s << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << endl;
	s << "<net name=\"" << n.m_name << "\" creator=\"" << n.m_creator << "\"" << endl;
	s << "\t<info> " << n.m_info << " </info>" << endl;
	for (signed int i=0; i < n.m_plane.size(); i++)
	{
		s << n.m_plane[i]->toString();
	}
	s << "</net>" << endl;
	return s;
}

/*! The operator is not implemented yet */
istream& operator>> (istream& s, CvConvNet& n)
{
	cerr << "*** ERR: operator >> is not implemented yet" << endl;
}
