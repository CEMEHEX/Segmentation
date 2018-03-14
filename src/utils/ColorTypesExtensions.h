#ifndef COLOR_TYPES_EXTENSIONS_H
#define COLOR_TYPES_EXTENSIONS_H

namespace std {
template<>
class hash<CvScalar> {
public:
    size_t operator()(const CvScalar &cvScalar) const
    {
        const size_t prime = 31;
        size_t res = 1;

        size_t h1 = std::hash<double>()(cvScalar.val[0]);
        size_t h2 = std::hash<double>()(cvScalar.val[1]);
        size_t h3 = std::hash<double>()(cvScalar.val[2]);
        size_t h4 = std::hash<double>()(cvScalar.val[3]);

        res = prime * res + (h1 ^ (h1 >> 32));
        res = prime * res + (h2 ^ (h2 >> 32));
        res = prime * res + (h3 ^ (h3 >> 32));
        res = prime * res + (h4 ^ (h4 >> 32));

        return res;
    }
};
}

bool operator==(const CvScalar& lhs, const CvScalar& rhs);

CvScalar getColor(cv::Mat& img, int i, int j);

cv::Vec3b cvScalar2Vec3b(const CvScalar& sc);

#endif // COLOR_TYPES_EXTENSIONS_H
