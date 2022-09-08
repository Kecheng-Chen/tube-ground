#include <cmath>

#include <algorithm>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <sstream>
#include <vector>


//! Mesh reader class
//! \brief Mesh class for creating nodes and elements from GMSH input
class MeshReader {
 public:

  //! \brief Read keywords
  //! \param[in] file Input file stream
  //! \param[in] keyword Search for keyword

  // check if contain keywords
  void read_keyword(std::ifstream& file, const std::string& keyword) {

    bool read_status = false;
    std::string line;
    file.clear();
    file.seekg(0, std::ios::beg);
    while (std::getline(file, line)) {
      if (line != keyword) {
        if (line.find(keyword) != std::string::npos) {
          break;
        };
      } else {
        std::cout << "Read keyword: " << keyword << " successfully\n";
        read_status = true;
        break;
      }
    }
    if (!read_status) {
      std::cerr << "Cannot find keyword: " << keyword << '\n';
    }
  }

  //! \brief Read ids and coordinates of vertices
  //! \param[in] file Input file stream object of msh file

  // read all nodes
  // related variable: vertices_ (array of vector)
  void read_vertices(std::ifstream& file) {
    read_keyword(file, "$Nodes");

    std::string line;
    std::getline(file, line);
    std::istringstream istream(line);

    // Total number of vertices
    unsigned nvertices;
    istream >> nvertices;
    std::cout << "Total number of vertices = " << nvertices << '\n';

    // Vertex id and coordinates
    unsigned vid = std::numeric_limits<unsigned>::max();
    std::vector<double> coordinates(3, 0);

    // Iterate through all vertices in the mesh file
    for (unsigned i = 0; i < nvertices;) {
      std::getline(file, line);
      std::istringstream istream(line);
      if (line.find('#') == std::string::npos && line != "") {
        // Initialise ids and coordinates
        vid = std::numeric_limits<unsigned>::max();

        // Read ids and coordinates
        istream >> vid;
        for (unsigned j = 0; j < coordinates.size(); ++j)
          istream >> coordinates.at(j);

        // Add vertex coordinates and id to a map
        vertices_[vid] = coordinates;

        // Increament number of vertex on successful read
        ++i;
      } else {
        std::cerr << "Invalid entry for node: " << line << '\n';
      }
    }
  }

  //! \brief Read ids and coordinates of vertices
  //! \param[in] file Input file stream object of msh file

  // read node on interior line and erase the duplicate (read txt file)
  // iline_vert_ids_: node ids
  void read_interior_line(std::ifstream& file) {

    std::string line;
    std::getline(file, line);
    std::istringstream istream(line);

    // Total number of vertices
    unsigned nlines;
    istream >> nlines;
    std::cout << "Total number of interior lines = " << nlines << '\n';

    // line ID
    unsigned lid = std::numeric_limits<unsigned>::max();
    //! Element type
    unsigned element_type = std::numeric_limits<unsigned>::max();
    //! Number of tags
    unsigned ntags = std::numeric_limits<unsigned>::max();
    unsigned tag = std::numeric_limits<unsigned>::max();
    //! Object id
    unsigned object_id = std::numeric_limits<unsigned>::max();
    //! Node id
    unsigned node_id = std::numeric_limits<unsigned>::max();

    for (unsigned i = 0; i < nlines; ++i) {
      std::getline(file, line);
      std::istringstream istream(line);
      if (line.find('#') == std::string::npos && line != "") {
        // Read ids and element type
        istream >> lid;
        istream >> element_type;
        istream >> ntags;
        istream >> object_id;
        // Read element tags
        for (unsigned j = 0; j < ntags - 1; ++j) {
          istream >> tag;
        }

        for (unsigned id = 0; id < 2; ++id) {
          istream >> node_id;
          iline_vert_ids_.push_back(node_id);
        }

      } else {
        //std::cerr << "Invalid entry for node: " << line << '\n';
      }
    }
    // Remove duplicates
    std::sort(iline_vert_ids_.begin(), iline_vert_ids_.end());
    iline_vert_ids_.erase(std::unique(iline_vert_ids_.begin(), iline_vert_ids_.end()), iline_vert_ids_.end());

/*
std::cout<<"vert id on interior line: ";
for(auto id : iline_vert_ids_) std::cout<<id<<", ";
std::cout<<std::endl;
std::cout<<"number of vertices: "<<vertices_.size()<<std::endl;
*/

  }

  //! \brief Check if a msh file exists
  //! \param[in] filename Mesh file name

  // call "read_vertices" and "read_interior_line"
  void read_mesh(const std::string& mesh_filename, const std::string& line_filename) {
    std::ifstream mesh_file;
    mesh_file.open(mesh_filename.c_str(), std::ios::in);
    if (!mesh_file.is_open())
      throw std::runtime_error("Specified GMSH file does not exist");
    if (mesh_file.good()) {
      read_vertices(mesh_file);
    }
    mesh_file.close();

    std::ifstream line_file;
    line_file.open(line_filename.c_str(), std::ios::in);
    if (!line_file.is_open())
      throw std::runtime_error("Specified txt file does not exist");
    if (line_file.good()) {
      read_interior_line(line_file);
    }
    line_file.close();
  }

  //! Return coords of vertices on interior line

  // return vector of interior line vertices coordinate
  // related variable: interior_line_vertices
  std::vector<std::vector<double>> interior_line_vertices() {
    std::vector<std::vector<double>> interior_line_vertices;
    for(auto vert_id : iline_vert_ids_) interior_line_vertices.push_back(vertices_.at(vert_id));
    return interior_line_vertices;
  }

  //! \brief Read ids and coordinates of vertices
  //! \param[in] file Input file stream object of msh file

  // line_tag_map: pair<vertices ids, object_id>
  // face_tag_map: pair<vertices ids, object_id>
  // lines_, faces_: maps
  void read_line_and_face(std::ifstream& file) {
    read_keyword(file, "$Elements");

    std::string line;
    std::getline(file, line);
    std::istringstream istream(line);

    // Total number of elements
    unsigned nelements;
    istream >> nelements;
    std::cout << "Total number of elements = " << nelements << '\n';

    //! Element ID
    unsigned element_id = std::numeric_limits<unsigned>::max();
    //! Element type
    unsigned element_type = std::numeric_limits<unsigned>::max();
    //! Number of nodes in an element
    unsigned nnodes = std::numeric_limits<unsigned>::max();
    //! Number of tags
    unsigned ntags = std::numeric_limits<unsigned>::max();
    unsigned tag = std::numeric_limits<unsigned>::max();
    //! Node id
    unsigned node_id = std::numeric_limits<unsigned>::max();
    //! Object id
    unsigned object_id = std::numeric_limits<unsigned>::max();

    // Iterate through all vertices in the mesh file
    for (unsigned i = 0; i < nelements; ++i) {
      std::getline(file, line);
      std::istringstream istream(line);
      if (line.find('#') == std::string::npos && line != "") {
        // Read ids and element type
        istream >> element_id;
        istream >> element_type;
        istream >> ntags;
        istream >> object_id;

        // Break after reading line and face elements
        if(element_type >= 4) break;

        // Read element tags
        for (unsigned j = 0; j < ntags - 1; ++j) {
          istream >> tag;
        }

        // Find the number of vertices for an element type
        unsigned nvertices = map_element_type_vertices_.at(element_type);

        // vertex ids
        std::vector<unsigned> vertices;
        vertices.clear();

        for (unsigned id = 0; id < nvertices; ++id) {
          istream >> node_id;
          vertices.push_back(node_id);
        }

        // Store line and face elements
        if (element_type == 1) {
          std::pair<std::vector<unsigned>, unsigned> line_tag_map = std::make_pair(vertices, object_id);
          lines_.emplace(line_tag_map);
        } else if (element_type == 2 || element_type == 3) {
          std::pair<std::vector<unsigned>, unsigned> face_tag_map = std::make_pair(vertices, object_id);
          faces_.emplace(face_tag_map);
        }
      
      } else {
        std::cerr << "Invalid entry for node: " << line << '\n';
      }
    }

std::cout<<"Finish reading line and face element, number of lines: "<<lines_.size()<<", number of faces: "<<faces_.size()<<std::endl;

  }

  //! \brief Read ids and coordinates of vertices
  //! \param[in] file Input file stream object of msh file

  // tag_vert_ids_map: map<each line tag, vertices ids>
  // remove duplicates
  // iline_vertices_: map<each line tag, vertices coords>
  void add_internal_line_vertices(std::vector<unsigned> iline_tags) {
    iline_vertices_.clear();
    // Map of tag and corresponding vertex ids
    std::map<unsigned, std::vector<unsigned>> tag_vert_ids_map;
    // Loop through line elements to find those in interior boundaries
    for(const auto line_tag : lines_) {
      for(const auto iline_tag : iline_tags) {
        if(line_tag.second == iline_tag) {
          for(const auto vert_id : line_tag.first) tag_vert_ids_map[iline_tag].push_back(vert_id);
          break;
        }
      }
    }
    // Remove duplicates
    for(auto& tag_vert_ids : tag_vert_ids_map) {
      std::sort(tag_vert_ids.second.begin(), tag_vert_ids.second.end());
      tag_vert_ids.second.erase(std::unique(tag_vert_ids.second.begin(), tag_vert_ids.second.end()), tag_vert_ids.second.end());
    }
    // Find corresponding vertex coordinates
    for(const auto& tag_vert_ids : tag_vert_ids_map) {
      for(auto vert_id : tag_vert_ids.second) iline_vertices_[tag_vert_ids.first].push_back(vertices_.at(vert_id));
    }

  }

  //! \brief Read ids and coordinates of vertices
  //! \param[in] file Input file stream object of msh file

  // tag_vert_ids_map: map<each face tag, vertices ids>
  // remove duplicates
  // iface_vertices_: map<each face tag, vertices coords>
  void add_internal_face_vertices(std::vector<unsigned> iface_tags) {
    iface_vertices_.clear();
    // Map of tag and corresponding vertex ids
    std::map<unsigned, std::vector<unsigned>> tag_vert_ids_map;
    // Loop through line elements to find those in interior boundaries
    for(const auto face_tag : faces_) {
      for(const auto iface_tag : iface_tags) {
        if(face_tag.second == iface_tag) {
          for(const auto vert_id : face_tag.first) tag_vert_ids_map[iface_tag].push_back(vert_id);
          break;
        }
      }
    }
    // Remove duplicates
    for(auto& tag_vert_ids : tag_vert_ids_map) {
      std::sort(tag_vert_ids.second.begin(), tag_vert_ids.second.end());
      tag_vert_ids.second.erase(std::unique(tag_vert_ids.second.begin(), tag_vert_ids.second.end()), tag_vert_ids.second.end());
    }
    // Find corresponding vertex coordinates
    for(const auto& tag_vert_ids : tag_vert_ids_map) {
      for(auto vert_id : tag_vert_ids.second) iface_vertices_[tag_vert_ids.first].push_back(vertices_.at(vert_id));
    }

  }

  //! \brief Read ids and coordinates of vertices
  //! \param[in] file Input file stream object of msh file

  // iline_vertices_and_length_: map<vertices coord, vertices length>
  // each segment's length is calculated
  void add_internal_line_vertices_and_length(std::vector<unsigned> iline_tags) {
    iline_vertices_.clear();
    // Map of tag and corresponding vertex ids
    std::map<unsigned, std::vector<unsigned>> tag_vert_ids_map;
    // Loop through line elements to find those in interior boundaries
    for(const auto line_tag : lines_) {
      for(const auto iline_tag : iline_tags) {
        if(line_tag.second == iline_tag) {
          for(const auto vert_id : line_tag.first) tag_vert_ids_map[iline_tag].push_back(vert_id);
          break;
        }
      }
    }
    // Remove duplicates
    for(auto& tag_vert_ids : tag_vert_ids_map) {
      std::sort(tag_vert_ids.second.begin(), tag_vert_ids.second.end());
      tag_vert_ids.second.erase(std::unique(tag_vert_ids.second.begin(), tag_vert_ids.second.end()), tag_vert_ids.second.end());
    }

    // Initialize line length of iline vertices
    for(const auto tag_vert_ids : tag_vert_ids_map) {
      for(const auto vert_id : tag_vert_ids.second) {
        vertices_length_[vert_id] = 0;
      }
    }
    // Compute length of each internal line segment, map to vert id
    for(const auto line_tag : lines_) {
      std::vector<double> coords1 = vertices_.at(line_tag.first.at(0));
      std::vector<double> coords2 = vertices_.at(line_tag.first.at(1));
      double seg_length = sqrt(pow(coords1.at(0) - coords2.at(0), 2) + pow(coords1.at(1) - coords2.at(1), 2) + pow(coords1.at(2) - coords2.at(2), 2));
      // accumulate segment length to vert ids
      vertices_length_[line_tag.first.at(0)] += seg_length/2;
      vertices_length_[line_tag.first.at(1)] += seg_length/2;
    }

    // Find corresponding vertex coordinates
    for(const auto& tag_vert_ids : tag_vert_ids_map) {
      for(auto vert_id : tag_vert_ids.second) {
        std::pair<std::vector<double>, double> vertices_and_length = std::make_pair(vertices_.at(vert_id), vertices_length_.at(vert_id));
        iline_vertices_and_length_[tag_vert_ids.first].emplace(vertices_and_length);
      }
    }
  }

  //! \brief mesh vertices and elements, and find interior line and face
  //! \param[in] filename Mesh file name

  // read internal line and face information
  void read_mesh_interior_boundaries(const std::string& mesh_filename, const std::vector<unsigned> iline_tags, const std::vector<unsigned> iface_tags, bool is_compute_length) {
    std::ifstream mesh_file;
    mesh_file.open(mesh_filename.c_str(), std::ios::in);
    if (!mesh_file.is_open())
      throw std::runtime_error("Specified GMSH file does not exist");
    if (mesh_file.good()) {
      read_vertices(mesh_file);
      read_line_and_face(mesh_file);
      if(!is_compute_length) add_internal_line_vertices(iline_tags);
      else add_internal_line_vertices_and_length(iline_tags);
      add_internal_face_vertices(iface_tags);
    }
    mesh_file.close();
  }

  //! Return coords of vertices on interior line with tag

  // vertices coord at tag
  std::vector<std::vector<double>> interior_line_vertices_at_tag(const unsigned tag) {
    bool is_tag = false;
    for(const auto& iline : iline_vertices_) {
      if(tag == iline.first) {
        is_tag = true;
        break;
      }
    }
    if(is_tag) return iline_vertices_.at(tag);
    else {
      std::cout<<"Input interior line tag: "<<tag<<" is not found"<<std::endl;
      throw std::runtime_error("Invalid interior line tag");
    }
  }

  //! Return coords and length of vertices on interior line with tag

  // vertices length at tag
  std::map<std::vector<double>, double> interior_line_vertices_length_at_tag(const unsigned tag) {
    bool is_tag = false;
    for(const auto& iline : iline_vertices_and_length_) {
      if(tag == iline.first) {
        is_tag = true;
        break;
      }
    }
    if(is_tag) return iline_vertices_and_length_.at(tag);
    else {
      std::cout<<"Input interior line tag: "<<tag<<" is not found"<<std::endl;
      throw std::runtime_error("Invalid interior line tag");
    }
  }

  //! Return coords of vertices on interior face with tag

  // for face, return indices coord at tag
  std::vector<std::vector<double>> interior_face_vertices_at_tag(const unsigned tag) {
    bool is_tag = false;
    for(const auto& iface : iface_vertices_) {
      if(tag == iface.first) {
        is_tag = true;
        break;
      }
    }
    if(is_tag) return iface_vertices_.at(tag);
    else {
      std::cout<<"Input interior face tag: "<<tag<<" is not found"<<std::endl;
      throw std::runtime_error("Invalid interior face tag");
    }
  }


  //! \brief Read C++ map from a text file
  //! \param[in] file Input file stream object of msh file

  // read map from txt
  std::map<unsigned, double> read_map(std::string filename, unsigned nlines) {

    std::map<unsigned, double> int_double_map;
    int_double_map.clear();

    std::ifstream file;
    file.open(filename.c_str(), std::ios::in);
    if (!file.is_open())
      throw std::runtime_error("Specified txt file for reading map does not exist");
    if (file.good()) {
      std::string line;

      // ID
      unsigned id = std::numeric_limits<unsigned>::max();
      //! Value
      double value = std::numeric_limits<unsigned>::max();

      for (unsigned i = 0; i < nlines; ++i) {
        std::getline(file, line);
        std::istringstream istream(line);
        // Read ids and element type
        istream >> id;
        istream >> value;
        int_double_map[id] = value;
      }
    }
    file.close();
    return int_double_map;
  }


 private:
  //! A map of vertex id and coordinates
  std::map<unsigned, std::vector<double>> vertices_;
  //! A map of line elements and their tags
  std::map<std::vector<unsigned>, unsigned> lines_;
  //! A map of face elements and their tags
  std::map<std::vector<unsigned>, unsigned> faces_;
  //! vector of indices of vector on the interior line
  std::vector<unsigned> iline_vert_ids_;
  //! vector of indices of vector on the interior face
  std::vector<unsigned> iface_vert_ids_;
  //! Vertices on interior line and corresponding tag
  std::map<unsigned, std::vector<std::vector<double>>> iline_vertices_;
  //! A map of vertex id and length of line belongs to it
  std::map<unsigned, double> vertices_length_;
  //! Vertices on interior line and length of line belong to it and corresponding tag
  std::map<unsigned, std::map<std::vector<double>, double>> iline_vertices_and_length_;
  //! Vertices on interior face and corresponding tag
  std::map<unsigned, std::vector<std::vector<double>>> iface_vertices_;
  //! Map of element type and number of vertices
  const std::map<unsigned, unsigned> map_element_type_vertices_{
      {1, 2},  // 2-node line.
      {2, 3},  // 3-node triangle.
      {3, 4},  // 4-node quadrangle.
      {4, 4},  // 4-node tetrahedron.
      {5, 8}   // 8-node hexahedron
  };
};