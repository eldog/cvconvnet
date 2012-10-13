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
 * \brief Implementation of default method for generic plane interface
 * \author Akhmed Umyarov
 * \date 2007
 */

#include "cvgenericplane.h"
#include <cassert>

using namespace std;

// Constructors/Destructors
//  
/*!
 * Base class constructor creates a new generic plane with specified name,
 * specified size of feature map and specified size of "neuron window"
 * \param id name of the plane
 * \param fmapsz size of the featuremap for this plane
 * \param neurosz size of "neuron window" (for instance 5x5 means that
 * we have a neuron that is connected to 25 outputs of his 
 * predecessor neuron feature map)
 */
CvGenericPlane::CvGenericPlane ( std::string id, CvSize fmapsz, CvSize neurosz )
{
	assert( fmapsz.width>0 && fmapsz.height>0 );
	assert( neurosz.width>=0 && neurosz.width>=0 );

	// Initialize parameters
	m_id = id;

	// Originally the plane is not connected to any other planes
	m_connected = 0;

	// Save plane topology 
	m_fmapsz = fmapsz;
	m_neurosz = neurosz;
	
	// Create null feature map
	m_fmap = cvCreateMat(fmapsz.height,fmapsz.width,CV_64FC1);
	assert( m_fmap != NULL );

	cvSetZero(m_fmap);
	
	m_weight = vector<double> ();
	m_pplane = vector<CvGenericPlane *> ();
}

CvGenericPlane::~CvGenericPlane ( ) 
{ 
	assert( m_fmap != NULL );
	cvReleaseMat( &m_fmap );
}

//  
// Methods
//  
/*!
 * The method connects the plane to the given parent planes
 * It also notifies each parent plane that this plane is their child plane
 * (will be used for bprop)
 * \param pplane parent planes
 * \return status
 */
int CvGenericPlane::connto(vector<CvGenericPlane *> &pplane)
{
	m_pplane = pplane;
	m_connected = 1;

	m_pfmap.resize(m_pplane.size());
	for (int i=0; i<m_pplane.size(); i++)
	{
		// Cache pointer to parents' fmaps
		m_pfmap[i] = m_pplane[i]->getfmap();

		// Connect as a child to a parent
		m_pplane[i]->connchild(this);				
	}

	return 1;
}

/*!
 * The method connects the plane to its child.
 * It is used only by connto() method in order to notify a parent
 * that it is being connected by a child.
 * After all connto() are done, each plane will have a list of parents
 * as well as list of its childs.
 */
int CvGenericPlane::connchild(CvGenericPlane *cplane)
{
	m_cplane.push_back( cplane );

	return 1;
}

/*!
 * The method disconnects the plane from all planes
 * \return status
 */
int CvGenericPlane::disconn()
{
	m_connected = 0;
	m_pplane.clear();
	m_cplane.clear();
	m_pfmap.clear();
	return 1;
}

/*!
 * The method explicitly sets the values of plane's feature map.
 * It just copies feature map given in the input parameter
 * to the plane's own storage.
 * \param source feature map to be copied from (CvMat or IplImage)
 * \return status
 */
int CvGenericPlane::setfmap ( CvArr * source ) 
{
	int width = cvGetSize(m_fmap).width;
	int height = cvGetSize(m_fmap).height;

	if ( (source == NULL) || !(cvGetSize(source).width == width 
		&& cvGetSize(source).height == height) )
		return 0;
	
	// Copy the image into matrix (and convert from bytes to doubles).
	cvConvertScale(source,m_fmap);
		
	return 1;
}

/*!
 * \return pointer to feature map of the plane
 */
CvMat * CvGenericPlane::getfmap ( ) 
{
	return m_fmap;
}

/*! The method explicitly sets the weights of the neuron
 */
int CvGenericPlane::setweight(std::vector<double> &weights)
{
	// Setting weights is only allowed when we are connected
	if (!m_connected) return 0;

	m_weight = weights;
	return 1;
}

/*!
 * \return string id of the plane
 */
string CvGenericPlane::getid()
{
	return m_id;
}
