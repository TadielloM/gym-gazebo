/*Test convert octomap_msgs with SWIG*/
%module conversions
%{
    // #include <octomap/octomap.h>
    #include <octomap_msgs/Octomap.h>
    // #include <octomap/ColorOcTree.h>
    #include "conversions.h"
    // #include <iostream>
    // #include <string>
    // #include <ros/ros.h>    // #include <memory>

%}

%include "typemaps.i"

%typemap(in) const octomap_msgs::Octomap& {
    // $1->header.seq = 213; 
    // PyTypeObject* type = $input->ob_type;
    // const char* p = type->tp_name;
    // std::cout << p <<std::endl;
    // std::string x = PyObject_HasAttrString($input, "header")?"SI ci e":"NON ci e";
    // std::cout << x << std::endl;
    // std::cout << PyString_AsString(PyObject_Str(PyObject_GetAttrString(PyObject_GetAttrString($input, "header"),"stamp"))) <<std::endl;
    octomap_msgs::Octomap tmp = octomap_msgs::Octomap();
    tmp.header.seq = PyInt_AsSsize_t(PyObject_Str(PyObject_GetAttrString(PyObject_GetAttrString($input, "header"),"seq")));
    tmp.header.frame_id = PyString_AsString(PyObject_Str(PyObject_GetAttrString(PyObject_GetAttrString($input, "header"),"frame_id")));
    // ros::Time stamp(PyLong_AsDouble(PyObject_Str(PyObject_GetAttrString(PyObject_GetAttrString($input, "header"),"stamp"))));
    // tmp.header.stamp = stamp;
    tmp.id = PyString_AsString(PyObject_Str(PyObject_GetAttrString($input, "id")));
    tmp.resolution = PyInt_AsLong(PyObject_Str(PyObject_GetAttrString(PyObject_GetAttrString($input, "header"),"seq")));
    std::vector<signed char> a1;
    
    // int data[PySequence_Length(PyObject_GetAttrString($input,"data"))];
    for(int i=0; i<PySequence_Length(PyObject_GetAttrString($input,"data")); i++){
        PyObject *o = PySequence_GetItem(PyObject_GetAttrString($input,"data"), i);
        if (PyNumber_Check(o)) {
            tmp.data.push_back((signed char) PyFloat_AsDouble(o));
        } else {
            PyErr_SetString(PyExc_ValueError, "Sequence elements must be numbers");      
            SWIG_fail;
        }
    }
    tmp.data = a1;
    $1 = &tmp;
} 


// %typemap(out) octomap::AbstractOcTree* {
    
// }


%include "conversions.h"    