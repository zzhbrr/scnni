/*
 * @Author: zzh
 * @Date: 2023-03-05
 * @LastEditTime: 2023-03-05 16:19:06
 * @Description: 
 * @FilePath: /SCNNI/src/operator.cpp
 */

#include "scnni/operator.hpp"
namespace scnni {
auto Parameter::InitFromString(const std::string &s) -> Parameter {
    Parameter p;
    p.type_ = 0;

    if (value == "None" || value == "()" || value == "[]")
    {
        return p;
    }

    if (value == "True" || value == "False")
    {
        // bool
        p.type = 1;
        p.b = value == "True";
        return p;
    }

    if (value[0] == '(' || value[0] == '[')
    {
        // list
        std::string lc = value.substr(1, value.size() - 2);
        std::istringstream lcss(lc);

        while (!lcss.eof())
        {
            std::string elem;
            std::getline(lcss, elem, ',');

            if ((elem[0] != '-' && (elem[0] < '0' || elem[0] > '9')) || (elem[0] == '-' && (elem[1] < '0' || elem[1] > '9')))
            {
                // string
                p.type = 7;
                p.as.push_back(elem);
            }
            else if (elem.find('.') != std::string::npos || elem.find('e') != std::string::npos)
            {
                // float
                p.type = 6;
                p.af.push_back(std::stof(elem));
            }
            else
            {
                // integer
                p.type = 5;
                p.ai.push_back(std::stoi(elem));
            }
        }
        return p;
    }

    if ((value[0] != '-' && (value[0] < '0' || value[0] > '9')) || (value[0] == '-' && (value[1] < '0' || value[1] > '9')))
    {
        // string
        p.type = 4;
        p.s = value;
        return p;
    }

    if (value.find('.') != std::string::npos || value.find('e') != std::string::npos)
    {
        // float
        p.type = 3;
        p.f = std::stof(value);
        return p;
    }

    // integer
    p.type = 2;
    p.i = std::stoi(value);
    return p;

}
} // namespace scnni