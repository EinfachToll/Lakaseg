#include <iostream>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <deque>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <csignal>

#ifdef _OPENMP
#include <omp.h>
#endif


// fremde Bibliotheken aus 3rd_party/ einbinden
#include "CImg/CImg.h"
using namespace cimg_library;

#include "SimpleJSON/src/JSON.cpp"
#include "SimpleJSON/src/JSONValue.cpp"

#include "maxflow-v3.04.src/graph_mf.h"
#include "maxflow-v3.04.src/graph_mf.cpp"
#include "maxflow-v3.04.src/maxflow.cpp"



// (fast) universelles Makro zum Debuggen mit D(variable);
#define D(x) do { std::cerr << "DEBUG: " << __FILE__ << "(" << __LINE__ << ") " << #x << " = ->" << x << "<-" << std::endl; } while(0)

// damit man auch einen std::vector ausgeben kann
template <typename T>
std::ostream& operator<<(std::ostream& out, std::vector<T>& vec)
{
    out << '[';
    for(size_t i = 0; i < vec.size(); ++i) {
        out << vec[i] << ' ';
    }
    out << ']';
    return out;
}



// Signal-Handler erstellen, der bei Strg+c das Programm beendet.
// Ist nötig, weil diese Bibliothek innerhalb vom Python-Prozess läuft, an den
// auch die Signale gehen würden.
void signal_handler(int signal_number) {
    std::cout << "Signal " << signal_number << " erhalten. Machs gut!" << std::endl;
    std::exit(1);
}

void install_signal_handler() {
    std::signal(SIGINT, signal_handler);
}



// Parameter, die während des Trainings oder der Inferenz verwendet werden.
// Es ist zwar ganz ganz schlechter Stil, globale Variablen daraus zu machen,
// aber die werden eben oft und über den ganzen Quelltext verteilt verwendet.
unsigned char WINDOW_SIZE;
unsigned char WINDOW_RADIUS;
unsigned short MAX_TREE_DEPTH;
unsigned int TESTOBJECT_TRIES;
unsigned short FOREST_SIZE;

double PAIRWISE_ENERGY;
double PAIRWISE_FACTOR;



// Hilfsfunktion, die ein Bild aus einer Datei lädt, aber nur den R-Kanal, weil
// wir annehmen, dass die hier verwendeten Bilder Grauwertbilder sind.
CImg<unsigned char>* load_one_channel(std::string filename) {
    CImg<unsigned char>* image_from_file;
    try {
        image_from_file = new CImg<unsigned char>(filename.c_str());
    } catch (...) {
        std::cerr << "Fehler: " << filename << " konnte nicht gelesen werden" << std::endl;
        std::exit(1);
    }

    if(image_from_file->spectrum() == 1) {
        return image_from_file;
    } else {
        CImg<unsigned char>* one_channel_image = new CImg<unsigned char>();
        (*one_channel_image) = image_from_file->get_channel(0);
        delete image_from_file;
        return one_channel_image;
    }
}



class TrainingData
{

public:

    std::vector<CImg<unsigned char>*> training_images;
    // Für jedes Trainingsbild eine Maske mit den Labels:
    // 1 ist Hintergrund, 2 ist Vordergrund, 0 wird beim Lernen ignoriert
    std::vector<CImg<unsigned char>*> label_masks;

    // die Farben für Vorder- und Hintergrund in den Ground-Truth-Bildern,
    // damit bei der Inferenz wieder Label-Bilder mit diesen Farben produziert
    // werden können
    unsigned char background_color;
    unsigned char foreground_color;

    unsigned int number_of_labeled_pixels;



    TrainingData(std::vector<std::string> training_image_filenames, std::vector<std::string> label_filenames) {

        if(training_image_filenames.size() != label_filenames.size()) {
            std::cerr << "Fehler: Ungleiche Anzahl von Trainings- und Labelbildern" << std::endl;
            std::exit(1);
        }

        this->background_color = 0;
        this->foreground_color = 0;
        this->number_of_labeled_pixels = 0;

        for(unsigned int i = 0; i < training_image_filenames.size(); ++i) {
            training_images.push_back(load_one_channel(training_image_filenames[i]));


            // das Bild mit den Labels laden und daraus die Maske erstellen

            CImg<unsigned char>* labels = load_one_channel(label_filenames[i]);

            if(training_images[i]->width() != labels->width() || training_images[i]->height() != labels->height()) {
                std::cerr << "Fehler: " << training_image_filenames[i] << " muss die gleiche Größe haben wie " << label_filenames[i] << std::endl;
                std::exit(1);
            }

            // die zwei in den Ground-Truth-Bildern vorkommenden Farben raussuchen
            cimg_forXY((*labels), x, y) {
                unsigned char c = (*labels)(x, y);
                if(c != 0) {
                    if(this->background_color == 0) {
                        this->background_color = c;
                    } else {
                        if(c != this->background_color && this->foreground_color == 0) {
                            this->foreground_color = c;
                            // Hässliches goto statt break, weil cimg_forXY ein
                            // Makro ist, das nach verschachtelten
                            // for-Schleifen expandiert. Ein break würde nur
                            // aus der innersten Schleife ausbrechen. Ich hatte
                            // schonmal Scherereien deswegen, darum verwende
                            // ich in diesen Schleifen immer goto statt break
                            goto break1;
                        }
                    }
                }
            }
break1:

            // wir gehen davon aus, dass der Vordergrund ein helleres Label als
            // der Hintergrund hat, deshalb wird der Wert hier ggf. vertauscht
            if(this->foreground_color < this->background_color) {
                unsigned char h = this->background_color;
                this->background_color = this->foreground_color;
                this->foreground_color = h;
            }


            CImg<unsigned char>* new_label_mask = new CImg<unsigned char>(labels->width(), labels->height(), 1, 1, 0);
            label_masks.push_back(new_label_mask);


            // da in den meisten Trainingsbildern deutlich mehr Vordergrund- als
            // Hintergrundpixel vorkommen dürften, werden von den
            // Vordergrundpixeln alle (mit 2) markiert (und damit beim Training
            // verwendet), aber von den Hintergrundpixeln werden nur welche
            // ausgewürfelt (so viele wie Vordergrundpixel)
            unsigned long number_of_foreground_pixels = 0;
            unsigned long number_of_background_pixels = 0;
            cimg_for_insideXY(*new_label_mask, x, y, WINDOW_RADIUS) {
                unsigned char c = (*labels)(x, y);

                if(c != 0) {
                    if(c == this->background_color) {
                        ++number_of_background_pixels;
                    } else {
                        if(c == this->foreground_color) {
                            (*new_label_mask)(x, y) = 2;
                            ++number_of_labeled_pixels;
                            ++number_of_foreground_pixels;
                        } else {
                            std::cout << "Es darf höchtens 2 verschiedene Labels geben" << std::endl;
                            exit(1);
                        }
                    }
                }
            }

            if(number_of_background_pixels > number_of_foreground_pixels) {

                // von den Hintergrundpixeln werden auch alle markiert, die
                // sich in einem gewissen Radius rund um Vordergrundgebiete
                // befinden, weil die möglicherweise etwas interessanter für
                // die Unterscheidung zwischen Vorder- und Hintergrund sind
                new_label_mask->dilate(15); // dehnt die Vordergrundpixel quasi aus
                cimg_for_insideXY(*new_label_mask, x, y, WINDOW_RADIUS) {
                    if((*new_label_mask)(x, y) == 2 && (*labels)(x, y) != this->foreground_color) {
                        (*new_label_mask)(x, y) = 1;
                        --number_of_foreground_pixels;
                        ++number_of_labeled_pixels;
                        if(number_of_foreground_pixels <= 0ul) {
                            goto break2;
                        }
                    }
                }
break2:


                // dann im Rest des Bildes zufällig Hintergrundpixel auswählen
                // und markieren, bis wir so viele wie Vordergrund markiert
                // haben
                for(unsigned long i = 0; i < number_of_foreground_pixels; ++i) {
                    while(true) {
                        int x = (std::rand() % (labels->width() - 2*WINDOW_RADIUS)) + WINDOW_RADIUS;
                        int y = (std::rand() % (labels->height() - 2*WINDOW_RADIUS)) + WINDOW_RADIUS;
                        if((*labels)(x, y) == this->background_color && (*new_label_mask)(x, y) == 0) {
                            (*new_label_mask)(x, y) = 1;
                            ++number_of_labeled_pixels;
                            break;
                        }
                    }
                }

            } else {  // wenn es doch mehr Vordergrund- als Hintergrundpixel gibt, werden auch alle Hintergrundpixel markiert
                cimg_forXY((*new_label_mask), x, y) {
                    if((*labels)(x, y) == this->background_color) {
                        (*new_label_mask)(x, y) = 1;
                        ++number_of_labeled_pixels;
                    }
                }
            }

            delete labels;
        }
    }


    ~TrainingData() {
        for(unsigned int i = 0; i < training_images.size(); ++i) {
            delete training_images[i];
            delete label_masks[i];
        }
    }
};



// jeder Blattknoten der Entscheidungsbäume hat einen Zeiger auf so ein
// LeafInfo-Objekt. Es enthält einfach die (empirische) Wahrscheinlichkeit,
// dass das Pixel, das an diesem Blattknoten ankommt, ein Vordergrundpixel ist
class LeafInfo
{

public:
    double foreground_probability;

    JSONValue* to_json()
    {
        return new JSONValue(foreground_probability);
    }

    static LeafInfo* from_json(JSONValue* value)
    {
        LeafInfo* new_leafinfo = new LeafInfo;
        new_leafinfo->foreground_probability = value->AsNumber();
        return new_leafinfo;
    }
};




// jeder innere Knoten eines Entscheidungsbaums hat einen Zeiger auf ein
// Testobjekt, das (mit der Funktion goes_left()) ein Pixel entweder nach links
// oder nach rechts im Entscheidungsbaum weiterschickt

// Hier werden 3 verschiedene Varianten solcher Klassen definiert, aber nur
// PixelDifferenceTest wird verwendet. Die anderen beiden kann man zum debuggen
// verwenden.


// so ein Testobjekt testet, ob der Unterschied der Grauwerte von zwei
// bestimmten Pixeln in der Nachbarschaft des zu klassifizierenden Pixels
// kleiner oder größer als ein Schwellwert ist
class PixelDifferenceTest
{
public:
    short offset_pixel1_x;
    short offset_pixel1_y;
    short offset_pixel2_x;
    short offset_pixel2_y;
    short difference_threshold;
    static std::wstring name;


    bool goes_left(CImg<unsigned char>* image, unsigned int x, unsigned int y)
    {
        return ((*image)(x + offset_pixel1_x, y + offset_pixel1_y) - (*image)(x + offset_pixel2_x, y + offset_pixel2_y)) < difference_threshold;
    }


    // ein Testobjekt wird erzeugt, indem einfach zufällig innerhalb kleinen
    // Fenster rund um ein Pixel zwei Nachbarpositionen und der Schwellwert
    // ausgewürfelt werden
    static PixelDifferenceTest* sample()
    {
        PixelDifferenceTest* testobject = new PixelDifferenceTest;
        testobject->offset_pixel1_x = (std::rand() % WINDOW_SIZE) - WINDOW_RADIUS;
        testobject->offset_pixel1_y = (std::rand() % WINDOW_SIZE) - WINDOW_RADIUS;
        testobject->offset_pixel2_x = (std::rand() % WINDOW_SIZE) - WINDOW_RADIUS;
        testobject->offset_pixel2_y = (std::rand() % WINDOW_SIZE) - WINDOW_RADIUS;
        testobject->difference_threshold = (std::rand() % 511) - 255;
        return testobject;
    }


    JSONValue* to_json()
    {
        JSONArray array;
        array.push_back(new JSONValue(static_cast<double>(offset_pixel1_x)));
        array.push_back(new JSONValue(static_cast<double>(offset_pixel1_y)));
        array.push_back(new JSONValue(static_cast<double>(offset_pixel2_x)));
        array.push_back(new JSONValue(static_cast<double>(offset_pixel2_y)));
        array.push_back(new JSONValue(static_cast<double>(difference_threshold)));
        return new JSONValue(array);
    }


    static PixelDifferenceTest* from_json(JSONValue* value)
    {
        JSONArray array = value->AsArray();
        PixelDifferenceTest* testobject = new PixelDifferenceTest;
        testobject->offset_pixel1_x = static_cast<short>(array[0]->AsNumber());
        testobject->offset_pixel1_y = static_cast<short>(array[1]->AsNumber());
        testobject->offset_pixel2_x = static_cast<short>(array[2]->AsNumber());
        testobject->offset_pixel2_y = static_cast<short>(array[3]->AsNumber());
        testobject->difference_threshold = static_cast<short>(array[4]->AsNumber());
        return testobject;
    }
};

// statische Member müssen in C++ immer außerhalb der Klasse definiert werden ...
std::wstring PixelDifferenceTest::name = L"PixelDifferenceTest";



/*
// zwei weitere Testfunktionen, die eventuell zum Debuggen nützlich sind

// so ein Testobjekt beachtet die Nachbarpixel überhaupt nicht, sondern testet
// einfach, ob das zu klassifizierende Pixel größer oder kleiner als ein
// Schwellwert ist
class PixelValueTest
{
public:
    unsigned char threshold;
    static std::wstring name;

    bool goes_left(CImg<unsigned char>* image, unsigned int x, unsigned int y)
    {
        return (*image)(x, y) < threshold;
    }

    static PixelValueTest* sample()
    {
        PixelValueTest* testobject = new PixelValueTest;
        testobject->threshold = std::rand() % 256;
        return testobject;
    }

    JSONValue* to_json()
    {
        return new JSONValue(static_cast<double>(threshold));
    }
};

std::wstring PixelValueTest::name = L"PixelValueTest";



// so ein Testobjekt testet, ob ein bestimmtes Pixel inder Nachbarschaft heller
// oder dunkler als ein Schwellwert ist
class AxisAlignedTest
{
public:

    short offset_x;
    short offset_y;
    unsigned char threshold;
    static std::wstring name;

    bool goes_left(CImg<unsigned char>* image, unsigned int x, unsigned int y)
    {
        return (*image)(x + offset_x, y + offset_y) < threshold;
    }

    static AxisAlignedTest* sample()
    {
        AxisAlignedTest* testobject = new AxisAlignedTest;
        testobject->offset_x = (std::rand() % WINDOW_SIZE) - WINDOW_RADIUS;
        testobject->offset_y = (std::rand() % WINDOW_SIZE) - WINDOW_RADIUS;
        testobject->threshold = std::rand() % 256;
        return testobject;
    }


    JSONValue* to_json()
    {
        JSONArray array;
        array.push_back(new JSONValue(static_cast<double>(offset_x)));
        array.push_back(new JSONValue(static_cast<double>(offset_y)));
        array.push_back(new JSONValue(static_cast<double>(threshold)));
        return new JSONValue(array);
    }

};

std::wstring AxisAlignedTest::name = L"AxisAlignedTest";

*/






// notwendige Forward declarations
template <typename T>
class Node;

template <typename T>
class Tree;


// während der Entscheidungsbaum aufgebaut wird, wird eine Liste von solchen
// LearningState-Objekten verwaltet
// So ein Objekt repräsentiert einen Knoten im Baum, für den bereits ein
// Testobjekt gelernt wurde, der aber noch keine Kindknoten hat
template <typename T>
struct LearningState
{
    Node<T>* node;

    // Tiefe des Knotens im Baum
    unsigned short depth;

    unsigned long from;
    unsigned long border;
    unsigned long to;
};





template <typename T>
class Node {

public:
    // linker und rechter Kindknoten
    Node* left_child;
    Node* right_child;

    // ein Knoten kann entweder ein innerer Knoten sein (dann hat er ein
    // test_object und leaf_info ist NULL) oder ein Blattknoten (dann
    // umgekehrt)
    T* test_object;
    LeafInfo* leaf_info;


    ~Node() {
        delete leaf_info;
        delete test_object;
        delete left_child;
        delete right_child;
    }


    // baut und gibt einen neuen Knoten zurück, der die Pixel am besten trennt
    // Das funktioniert, indem eine Anzahl von Testobjekten zufällig erzeugt
    // wird. Mit jedem davon trennt man die Trainingspixel, die an diesem
    // Knoten ankommen und misst in den entstandenen Teilmengen das Verhältnis
    // von Vordergrund- zu Hintergrundpixeln. Je ungleicher das Verhältnis,
    // desto besser. Das beste Trainingsobjekt wird für diesen Knoten genommen.
    static Node<T>* build_inner_node(TrainingData& labels, LearningState<T>& state, std::vector<unsigned int>& samples)
    {

        double lowest_expected_entropy = std::numeric_limits<double>::infinity();
        T* best_test = NULL;

        // wieviele Trainingsbeispiele das beste Testobjekt insgesamt nach
        // rechts bzw. links schickt und wieviele davon Vordergrundpixel sind
        unsigned long best_total_pixels_left = 0;
        unsigned long best_total_pixels_right = 0;
        unsigned long best_foreground_count_left = 0;
        unsigned long best_foreground_count_right = 0;

        // wird true, wenn dieser Knoten die Trainingsbeispiele perfekt trennt,
        // d.h. alle Pixel, die tatsächlich zum Vordergrund gehören, werden auf
        // die eine Seite und alle anderen auf die andere Seite geschickt
        bool low_entropy_left = false;
        bool low_entropy_right = false;


        for(unsigned int try_count = 0; try_count < TESTOBJECT_TRIES; ++try_count) {

            T* random_test_object = T::sample();

            // zählt, wieviele Trainingsbeispiele dieses Testobjekt nach links
            // bzw. rechts schickt und wieviele davon Vordergrundpixel sind
            unsigned long total_left = 0;
            unsigned long total_right = 0;
            unsigned long foreground_left = 0;
            unsigned long foreground_right = 0;

            // über alle Trainingsbeispiele iterieren, mit denen dieser Knoten
            // trainiert werden soll
            for(unsigned int i = state.from; i <= state.to; i += 3) {
                unsigned int idx = samples[i];
                unsigned int x = samples[i+1];
                unsigned int y = samples[i+2];

                if(random_test_object->goes_left(labels.training_images[idx], x, y)) {
                    // zählen, wieviele der Trainingspixel links landen und
                    // wieviele davon Vordergrundpixel sind. (Vordergrundpixel
                    // haben in der Maske den Wert 2, d.h. der Zähler wird hier
                    // um 1 hochgezählt)
                    foreground_left += (*(labels.label_masks[idx]))(x, y) - 1;
                    total_left++;
                } else {
                    foreground_right += (*(labels.label_masks[idx]))(x, y) - 1;
                    total_right++;
                }
            }

            // Wenn das Trainingsobjekt die Beispiele gar nicht trennt, sondern
            // alle auf eine Seite sortiert, samplen wir nochmal
            // TODO: gerät in eine Endlosschleife, wenn die Beispiele gar nicht
            // trennbar sind, z.B. weil sie identisch sind
            if(total_left == 0 || total_right == 0) {
                --try_count;
                delete random_test_object;
                continue;
            }


            // die Entropie (quasi die Ungleichverteilung) der Verteilung der
            // beiden Klassen VG und HG an den Ausgängen rechts und links
            // ausrechnen
            double entropy_left = 0.0;
            double entropy_right = 0.0;

            if(foreground_left > 0 && foreground_left < total_left) {
                double p = static_cast<double>(foreground_left) / total_left;
#ifdef _WIN32
                // unter Windows gibts die Funktion log2() nicht
                entropy_left = - ((p * log(p) + (1.0 - p) * log(1.0 - p)) / log(2.0));
#else
                entropy_left = - (p * log2(p) + (1.0 - p) * log2(1.0 - p));
#endif
            }
            if(foreground_right > 0 && foreground_right < total_right) {
                double p = static_cast<double>(foreground_right) / total_right;
#ifdef _WIN32
                entropy_right = - ((p * log(p) + (1.0 - p) * log(1.0 - p)) / log(2.0));
#else
                entropy_right = - (p * log2(p) + (1.0 - p) * log2(1.0 - p));
#endif
            }

            // daraus die durchschnittliche erwartete Entropie berechnen (ohne
            // zu normalisieren, weil wir nur das Minimum davon wollen)
            double expected_entropy = static_cast<double>(total_left) * entropy_left +
                static_cast<double>(total_right) * entropy_right;

            if(expected_entropy < lowest_expected_entropy) {
                lowest_expected_entropy = expected_entropy;
                delete best_test;
                best_test = random_test_object;
                // wenn rechts oder links nur Beispiele aus einer einzigen
                // Klasse ankommen, ist die Entropie dort 0. In diesem Fall
                // machen wir auf der Seite einen Blattknoten
                low_entropy_left = (entropy_left == 0.0);
                low_entropy_right = (entropy_right == 0.0);
                best_foreground_count_left = foreground_left;
                best_foreground_count_right = foreground_right;
                best_total_pixels_left = total_left;
                best_total_pixels_right = total_right;
            } else {
                delete random_test_object;
            }
        }


        Node<T>* new_node = new Node;
        new_node->leaf_info = NULL;
        new_node->left_child = NULL;
        new_node->right_child = NULL;
        new_node->test_object = best_test;


        // an new_node links einen Blattknoten anhängen, wenn die maximale
        // Tiefe erreicht ist oder die Entropie dort 0 ist
        if(low_entropy_left || state.depth >= MAX_TREE_DEPTH) {
            new_node->left_child = build_leaf_node(best_foreground_count_left, best_total_pixels_left);
        }

        // ebenso rechts
        if(low_entropy_right || state.depth >= MAX_TREE_DEPTH) {
            new_node->right_child = build_leaf_node(best_foreground_count_right, best_total_pixels_right);
        }

        return new_node;
    }


    static Node<T>* build_leaf_node(unsigned long foreground_count, unsigned long total)
    {
        Node<T>* new_leaf = new Node<T>;
        new_leaf->left_child = NULL;
        new_leaf->right_child = NULL;
        new_leaf->test_object = NULL;
        new_leaf->leaf_info = new LeafInfo;

        // foreground_probability ist einfach die Anzahl der ankommenden
        // Vordergrundpixel geteilt durch die Gesamtzahl der ankommenden Pixel
        new_leaf->leaf_info->foreground_probability = static_cast<double>(foreground_count) / static_cast<double>(total);

        return new_leaf;
    }


    static Node<T>* from_json(JSONValue* json_value)
    {
        Node<T>* new_node = new Node<T>;
        if(json_value->IsArray()) { // ein innerer Knoten
            JSONArray array = json_value->AsArray();
            new_node->test_object = T::from_json(array[0]);
            new_node->left_child = Node<T>::from_json(array[1]);
            new_node->right_child = Node<T>::from_json(array[2]);
            new_node->leaf_info = NULL;
        } else { // Blattknoten
            new_node->test_object = NULL;
            new_node->left_child = NULL;
            new_node->right_child = NULL;
            new_node->leaf_info = LeafInfo::from_json(json_value);
        }
        return new_node;
    }


    JSONValue* to_json()
    {
        if(test_object != NULL) {
            JSONArray node_array;
            node_array.push_back(test_object->to_json());
            node_array.push_back(left_child->to_json());
            node_array.push_back(right_child->to_json());
            return new JSONValue(node_array);
        } else {
            return leaf_info->to_json();
        }
    }

};



// sortiert die Liste im Intervall [from, to) so um, dass die Beispiele, die
// vom gegebenen Testobjekt nach links klassifiziert werden alle vor den
// anderen kommen. Gibt den Index mit der Grenze zurück, nämlich dem ersten von
// den rechten Beispielen.
template <typename T>
unsigned int rearrange_samples(std::vector<unsigned int>& samples, unsigned int from, unsigned int to, T* testobject, TrainingData& training)
{
    unsigned int left = from;
    unsigned int right = to;

    do {
        while(testobject->goes_left(training.training_images[samples[left]], samples[left+1], samples[left+2])) {
            left += 3;
        }

        while(!testobject->goes_left(training.training_images[samples[right]], samples[right+1], samples[right+2])) {
            right -= 3;
        }

        if(left >= right) {
            return left;
        }

        std::swap(samples[left], samples[right]);
        std::swap(samples[left+1], samples[right+1]);
        std::swap(samples[left+2], samples[right+2]);
        left += 3;
        right -= 3;

    } while(true);

    return left;
}



template <typename T>
class Tree
{

public:

    Node<T>* root;


    static Tree* train(TrainingData& labels)
    {
        Tree<T>* tree = new Tree;


        // eine Liste der Form [i1, x1, y1, i2, x2, y2, ...] mit Bildindex und
        // den Koordinaten aller Pixel im Trainingsbild, die für das Training
        // benutzt werden.
        std::vector<unsigned int> samples(3 * labels.number_of_labeled_pixels);
        unsigned int samples_count = 0;
        for(unsigned int i = 0; i < labels.label_masks.size(); ++i) {
            cimg_for_insideXY(*(labels.label_masks[i]), x, y, WINDOW_RADIUS) {
                if((*(labels.label_masks[i]))(x, y) > 0) {
                    samples[samples_count] = i;
                    samples[samples_count + 1] = x;
                    samples[samples_count + 2] = y;
                    samples_count += 3;
                }
            }
        }
        samples_count -= 3;


        // Wurzelknoten bauen, und zwar mit allen Samples aus der Liste
        LearningState<T> root_state;
        root_state.depth = 1;
        root_state.from = 0;
        root_state.to = samples_count;
        tree->root = Node<T>::build_inner_node(labels, root_state, samples);

        // die Liste so umsortieren, dass alle Pixel, die der Wurzelknoten nach
        // links schickt auch links in der Liste sitzen
        root_state.border = rearrange_samples(samples, root_state.from, root_state.to, tree->root->test_object, labels);


        // hier sind die Knoten drin, die schon ein Testobjekt haben, aber noch
        // keine Kindknoten
        std::deque<LearningState<T> > pending_nodes;
        LearningState<T> start_state;
        start_state.node = tree->root;
        start_state.depth = 1;
        start_state.from = 0;
        start_state.to = samples_count;
        start_state.border = root_state.border;

        pending_nodes.push_back(start_state);

        while(!pending_nodes.empty()) {
            LearningState<T> current_pending_node = pending_nodes.back();
            pending_nodes.pop_back();

            // prüfen, ob links nicht schon ein Blattknoten ist (z.B. weil die
            // maximale Tiefe erreicht wurde)
            if(current_pending_node.node->left_child == NULL) {
                LearningState<T> left_state = current_pending_node; // kopieren
                left_state.depth += 1;
                // der zukünftige linke Kindknoten soll nur die Trainingspixel
                // verwenden, die current_pending_node.node nach links schickt
                left_state.to = left_state.border - 3;
                Node<T>* new_node = Node<T>::build_inner_node(labels, left_state, samples);
                left_state.border = rearrange_samples(samples, left_state.from, left_state.to, new_node->test_object, labels);
                left_state.node->left_child = new_node;
                left_state.node = new_node;
                pending_nodes.push_back(left_state);
            }

            if(current_pending_node.node->right_child == NULL) {
                LearningState<T> right_state = current_pending_node;
                right_state.depth += 1;
                right_state.from = right_state.border;
                Node<T>* new_node = Node<T>::build_inner_node(labels, right_state, samples);
                right_state.border = rearrange_samples(samples, right_state.from, right_state.to, new_node->test_object, labels);
                right_state.node->right_child = new_node;
                right_state.node = new_node;
                pending_nodes.push_back(right_state);
            }
        }

        return tree;
    }


    ~Tree() {
        delete this->root;
    }


    static Tree* from_json(JSONValue* json_value)
    {
        Tree<T>* tree = new Tree;
        tree->root = Node<T>::from_json(json_value);
        return tree;
    }


    JSONValue* to_json()
    {
        return this->root->to_json();
    }


    LeafInfo* inference(CImg<unsigned char>& image, unsigned int x, unsigned int y)
    {
        Node<T>* current_node = this->root;
        while(current_node->test_object != NULL) {
            if(current_node->test_object->goes_left(&image, x, y))
                current_node = current_node->left_child;
            else
                current_node = current_node->right_child;
        }

        return current_node->leaf_info;
    }
};









template <typename T>
class Forest
{
public:
    std::vector<Tree<T>*> trees;
    unsigned char background_color;
    unsigned char foreground_color;


    static Forest<T> train(std::vector<std::string> training_image_filenames, std::vector<std::string> label_filenames) {

        Forest forest;

#pragma omp parallel for
        for(short i = 0; i < FOREST_SIZE; ++i) {

            // Die Konsolenausgabe ist nicht threadsicher, deshalb ist das ein
            // kritischer Abschnitt, d.h. solange ein Thread diese Codezeile
            // ausführt, darf die kein anderer auch ausführen. Andererseits
            // kann es auch lustig aussehen, wenn die Ausgabe
            // durcheinandergerät.
#pragma omp critical(output)
            std::cout << "Trainiere Baum " << i+1 << " von " << FOREST_SIZE << std::endl;

            // man muss für jeden Baum ein neues TrainingData-Objekt erstellen,
            // damit die Hintergrundpixel, die für das Training verwendet
            // werden, bei jedem Baum neu ausgewürfelt werden
            //
            // TODO: andere Lösung überlegen, das braucht bei mehreren Threads
            // unnötig Speicher, weil das Objekt auch die Trainingsbilder
            // enthält, die aber immer gleich sind
            TrainingData labels(training_image_filenames, label_filenames);

            Tree<T>* t = Tree<T>::train(labels);

            // da die STL nicht threadsicher ist, ist das Hinzufügen zu einem
            // Vektor auch ein kritischer Abschnitt
#pragma omp critical(append_to_list)
            forest.trees.push_back(t);
        }

        TrainingData labels(training_image_filenames, label_filenames); // nochmal so ein Objekt erstellen, um an die Vordergrund- und Hintergrundfarben zu kommen. TODO: was eleganteres überlegen
        forest.background_color = labels.background_color;
        forest.foreground_color = labels.foreground_color;

        return forest;
    }


    ~Forest<T>() {
        for(size_t i = 0; i < trees.size(); ++i) {
            delete trees[i];
        }
    }


    // gibt für ein Pixel die Wahrscheinlichkeit zurück, dass es sich um ein
    // Vordergrundpixel handelt
    double inference(CImg<unsigned char>& image, unsigned int x, unsigned int y) {
        double sum_foreground_probability = 0.0;
        for(unsigned int i = 0; i < FOREST_SIZE; ++i) {
            LeafInfo* leaf = this->trees[i]->inference(image, x, y);
            sum_foreground_probability += leaf->foreground_probability;
        }
        return sum_foreground_probability / FOREST_SIZE;
    }


    // Inferenz mit dem Maxflow-Algorithmus. Der Quelltext befindet sich in 3rd_party/maxflow-v3.04.src/
    CImg<unsigned char>* inference_maxflow(CImg<unsigned char>& image, const char* intermediate_result)
    {
        typedef Graph_mf<double, double, double> GraphType;

        // die Variablen im Graph sind alle Pixel außer die am Rand, weil für
        // die keine Ausgabe aus dem Random Forest als Unary Potential zur
        // Verfügung steht
        int grid_width = image.width() - 2*WINDOW_RADIUS;
        int grid_height = image.height() - 2*WINDOW_RADIUS;
        GraphType* graph = new GraphType(grid_width*grid_height, 2*grid_width*grid_height - grid_width - grid_height);

        CImg<unsigned char>* result = new CImg<unsigned char>(image.width(), image.height(), 1, 1, 0);
        int node_index = 0;
        cimg_for_insideXY(image, x, y, WINDOW_RADIUS) {

            double foreground_probability = inference(image, x, y);

            // Wahrscheinlichkeiten nahe bei 0 oder 1 sind erstens
            // unrealistisch und zweitens wird die Berechnung instabil
            if(foreground_probability < 0.0001) {
                foreground_probability = 0.0001;
            } else if(foreground_probability > 0.9999) {
                foreground_probability = 0.9999;
            }

            if(intermediate_result != NULL) {
                (*result)(x, y) = static_cast<unsigned char>(255 * foreground_probability);
            }

            graph->add_node();
            graph->add_tweights(node_index, -log(foreground_probability), -log(1.0 - foreground_probability));

            ++node_index;
        }

        if(intermediate_result != NULL) {
            result->save(intermediate_result);
        }

        // Energien zwischen Variablen (also zwischen benachbarten Pixeln) angeben
        for(int i = 0; i < grid_width*grid_height; ++i) {
            if((i + 1) % grid_width != 0) {  // alle Knoten außer die am rechten Rand
                graph->add_edge(i, i+1, PAIRWISE_ENERGY, PAIRWISE_ENERGY);
            }
            if(i < grid_width*(grid_height-1)) {  // alle Knoten außer die am unteren Rand
                graph->add_edge(i, i+grid_width, PAIRWISE_ENERGY, PAIRWISE_ENERGY);
            }
        }

        graph->maxflow();

        node_index = 0;
        cimg_for_insideXY(*result, x, y, WINDOW_RADIUS) {
            (*result)(x, y) = graph->what_segment(node_index) == GraphType::SOURCE ? this->background_color : this->foreground_color;
            ++node_index;
        }

        delete graph;

        return result;
    }


    unsigned char sample_corner_variable(unsigned char neighbor1_state, unsigned char neighbor2_state, float unary_pot) {
        // Produkt aller Faktoren, wenn der Zustand der betrachteten Variablen 0 ist
        double a = unary_pot;
        if(neighbor1_state)
            a *= PAIRWISE_FACTOR;
        if(neighbor2_state)
            a *= PAIRWISE_FACTOR;

        // analog für 1
        double b = (1.0 - unary_pot);
        if(!neighbor1_state)
            b *= PAIRWISE_FACTOR;
        if(!neighbor2_state)
            b *= PAIRWISE_FACTOR;

        // normalisieren
        a = a/(a+b);

        // und sampeln
#ifdef _WIN32
        return (double) rand() / (double) RAND_MAX > a;
#else
        return drand48() > a;
#endif
    }

    unsigned char sample_edge_variable(unsigned char neighbor1_state, unsigned char neighbor2_state, unsigned char neighbor3_state, float unary_pot) {
        double a = unary_pot;
        if(neighbor1_state)
            a *= PAIRWISE_FACTOR;
        if(neighbor2_state)
            a *= PAIRWISE_FACTOR;
        if(neighbor3_state)
            a *= PAIRWISE_FACTOR;

        double b = (1.0 - unary_pot);
        if(!neighbor1_state)
            b *= PAIRWISE_FACTOR;
        if(!neighbor2_state)
            b *= PAIRWISE_FACTOR;
        if(!neighbor3_state)
            b *= PAIRWISE_FACTOR;

        a = a/(a+b);

#ifdef _WIN32
        return (double) rand() / (double) RAND_MAX > a;
#else
        return drand48() > a;
#endif
    }

    unsigned char sample_inner_variable(unsigned char neighbor1_state, unsigned char neighbor2_state, unsigned char neighbor3_state, unsigned char neighbor4_state, float unary_pot) {
        double a = unary_pot;
        if(neighbor1_state)
            a *= PAIRWISE_FACTOR;
        if(neighbor2_state)
            a *= PAIRWISE_FACTOR;
        if(neighbor3_state)
            a *= PAIRWISE_FACTOR;
        if(neighbor4_state)
            a *= PAIRWISE_FACTOR;

        double b = (1.0 - unary_pot);
        if(!neighbor1_state)
            b *= PAIRWISE_FACTOR;
        if(!neighbor2_state)
            b *= PAIRWISE_FACTOR;
        if(!neighbor3_state)
            b *= PAIRWISE_FACTOR;
        if(!neighbor4_state)
            b *= PAIRWISE_FACTOR;

        a = a/(a+b);

#ifdef _WIN32
        return ((double) rand() / (double) RAND_MAX) > a;
#else
        return drand48() > a;
#endif
    }


    // Inferenz mit dem Gibbs-Sampling-Algorithmus
    CImg<unsigned char>* inference_gibbs(CImg<unsigned char>& image, const char* intermediate_result)
    {
        int grid_width = image.width() - 2*WINDOW_RADIUS;
        int grid_height = image.height() - 2*WINDOW_RADIUS;

        CImg<float>* unary_pots = new CImg<float>(image.width(), image.height(), 1, 1, 0);
        cimg_for_insideXY(image, x, y, WINDOW_RADIUS) {
            double foreground_probability = inference(image, x, y);

            if(foreground_probability < 0.0001) {
                foreground_probability = 0.0001;
            } else if(foreground_probability > 0.9999) {
                foreground_probability = 0.9999;
            }

            (*unary_pots)(x, y) = foreground_probability;
        }

        if(intermediate_result != NULL) {
            CImg<unsigned char> intermediate(image.width(), image.height(), 1, 1, 0);
            cimg_for_insideXY(intermediate, x, y, WINDOW_RADIUS) {
                intermediate(x, y) = static_cast<unsigned char>((*unary_pots)(x, y) * 255);
            }
            intermediate.save(intermediate_result);
        }


        // zu sampelnde Belegung von Zeit t
        CImg<unsigned char>* y_t = new CImg<unsigned char>(grid_width, grid_height, 1, 1, 0);

        // zählt für jedes Pixel, wie oft eine 1 gesampelt wurde
        CImg<unsigned int> count_ones(grid_width, grid_height, 1, 1, 0);

        // Anfangsbelegung zufällig initialisieren (alles 0 oder 1)
        cimg_forXY((*y_t), x, y) {
            (*y_t)(x, y) = rand() % 2;
        }

        const unsigned int N = 2000; // die Anzahl der Samplingschritte
        for(unsigned int versuch = 0; versuch < N+10; ++versuch) {
            if(versuch % 100 == 0) {
                std::cout << "Sampling-Schritt " << versuch << " von " << N << std::endl;
            }

            // die Variablen in y_t einzeln sampeln, basierend auf den
            // aktuellen Belegungen der Nachbarvariablen

            // erst die 4 Ecken
            (*y_t)(0, 0) = sample_corner_variable((*y_t)(1, 0), (*y_t)(0, 1), (*unary_pots)(0+WINDOW_RADIUS, 0+WINDOW_RADIUS));
            (*y_t)(grid_width-1, 0) = sample_corner_variable((*y_t)(grid_width-2, 0), (*y_t)(grid_width-1, 1), (*unary_pots)(grid_width-1+WINDOW_RADIUS, 0+WINDOW_RADIUS));
            (*y_t)(0, grid_height-1) = sample_corner_variable((*y_t)(0, grid_height-2), (*y_t)(1, grid_height-1), (*unary_pots)(0+WINDOW_RADIUS, grid_height-1+WINDOW_RADIUS));
            (*y_t)(grid_width-1, grid_height-1) = sample_corner_variable((*y_t)(grid_width-2, grid_height-1), (*y_t)(grid_width-1, grid_height-2), (*unary_pots)(grid_width-1+WINDOW_RADIUS, grid_height-1+WINDOW_RADIUS));

            // dann die Kanten links und rechts
            for(int y = 1; y < grid_height-1; ++y) {
                (*y_t)(0, y) = sample_edge_variable((*y_t)(0, y-1), (*y_t)(0, y+1), (*y_t)(1, y), (*unary_pots)(0+WINDOW_RADIUS, y+WINDOW_RADIUS));
                (*y_t)(grid_width-1, y) = sample_edge_variable((*y_t)(grid_width-1, y-1), (*y_t)(grid_width-1, y+1), (*y_t)(grid_width-2, y), (*unary_pots)(grid_width-1+WINDOW_RADIUS, y+WINDOW_RADIUS));
            }

            // Kanten oben und unten
            for(int x = 1; x < grid_width-1; ++x) {
                (*y_t)(x, 0) = sample_edge_variable((*y_t)(x-1, 0), (*y_t)(x+1, 0), (*y_t)(x, 1), (*unary_pots)(x+WINDOW_RADIUS, 0+WINDOW_RADIUS));
                (*y_t)(x, grid_height-1) = sample_edge_variable((*y_t)(x-1, grid_height-1), (*y_t)(x+1, grid_height-1), (*y_t)(x, grid_height-2), (*unary_pots)(x+WINDOW_RADIUS, grid_height-1+WINDOW_RADIUS));
            }

            // Variablen innen
            for(int x = 1; x < grid_width-1; ++x) {
                for(int y = 1; y < grid_height-1; ++y) {
                    (*y_t)(x, y) = sample_inner_variable((*y_t)(x-1, y), (*y_t)(x+1, y), (*y_t)(x, y-1), (*y_t)(x, y+1), (*unary_pots)(x+WINDOW_RADIUS, y+WINDOW_RADIUS));
                }
            }

            // Zähler erhöhen, wenns eine 1 ergibt
            // aber erst ab dem 10. Durchlauf, weil sich die Markow-Kette erst einschwingen muss
            if(versuch >= 10) {
                count_ones += (*y_t);
            }
        }

        delete y_t;
        delete unary_pots;

        CImg<unsigned char>* result = new CImg<unsigned char>(image.width(), image.height(), 1, 1, 0);
        cimg_for_insideXY((*result), x, y, WINDOW_RADIUS) {
            (*result)(x, y) = (count_ones(x-WINDOW_RADIUS, y-WINDOW_RADIUS) > (N/2) ? this->background_color : this->foreground_color);
        }

        return result;
    }



    void write_to_file(std::string filename)
    {
        JSONArray json_root;

        JSONObject learning_parameters;
        learning_parameters[L"Test Type"] = new JSONValue(T::name);
        learning_parameters[L"Max tree depth"] = new JSONValue(static_cast<double>(MAX_TREE_DEPTH));
        learning_parameters[L"Testobject tries"] = new JSONValue(static_cast<double>(TESTOBJECT_TRIES));
        learning_parameters[L"Forest size"] = new JSONValue(static_cast<double>(FOREST_SIZE));
        learning_parameters[L"Window radius"] = new JSONValue(static_cast<double>(WINDOW_RADIUS));

        json_root.push_back(new JSONValue(learning_parameters));

        json_root.push_back(new JSONValue(static_cast<double>(this->background_color)));
        json_root.push_back(new JSONValue(static_cast<double>(this->foreground_color)));

        for(typename std::vector<Tree<T>*>::iterator it = trees.begin(); it != trees.end(); ++it) {
            json_root.push_back((*it)->to_json());
        }

        JSONValue *root_value = new JSONValue(json_root);

        std::wofstream out(filename.c_str());
        // Stringify(true) würde das ganze etwas lesefreundlicher ausgeben
        out << root_value->Stringify(false) << '\n';

        delete root_value;
    }


    static Forest<T> load_from_file(std::string filename)
    {
        std::wostringstream st;
        std::wifstream in(filename.c_str());
        if(!in) {
            std::cerr << "Fehler: " << filename << " konnte nicht gelesen werden" << std::endl;
            std::exit(1);
        }
        st << in.rdbuf();
        std::wstring json_string = st.str();
        JSONValue *value = JSON::Parse(json_string.c_str());

        Forest forest;

        JSONArray root_array = value->AsArray();

        JSONObject learning_parameters = root_array.at(0)->AsObject();
        FOREST_SIZE = static_cast<unsigned short>(learning_parameters[L"Forest size"]->AsNumber());
        TESTOBJECT_TRIES = static_cast<unsigned int>(learning_parameters[L"Testobject tries"]->AsNumber());
        MAX_TREE_DEPTH = static_cast<unsigned short>(learning_parameters[L"Max tree depth"]->AsNumber());
        WINDOW_RADIUS = static_cast<unsigned char>(learning_parameters[L"Window radius"]->AsNumber());
        WINDOW_SIZE = (2*WINDOW_RADIUS)+1;

        forest.background_color = root_array[1]->AsNumber();
        forest.foreground_color = root_array[2]->AsNumber();

        for(unsigned int i = 3; i < root_array.size(); ++i) {
            forest.trees.push_back(Tree<T>::from_json(root_array[i]));
        }

        delete value;
        return forest;
    }

};


// zum testen; vergleicht das Ergebnis der Inferenz mit einem von Hand
// gelabelten Bild und schreibt das Resultat in eine Datei
static void print_result_statistics(const char* ground_truth_filename, CImg<unsigned char>* result)
{
    CImg<unsigned char> ground_truth;

    try{
        ground_truth.assign(ground_truth_filename);
    } catch (...) {
        std::cerr << "Fehler: " << ground_truth_filename << " konnte nicht gelesen werden" << std::endl;
        std::exit(1);
    }

    unsigned long number_of_labeled_pixels = 0;
    unsigned long number_of_correctly_labeled_pixels = 0;

    cimg_for_insideXY((*result), x, y, WINDOW_RADIUS) {
        unsigned char gt_label = ground_truth(x, y);
        if(gt_label > 0) {
            ++number_of_labeled_pixels;
            if((*result)(x, y) == gt_label) {
                ++number_of_correctly_labeled_pixels;
            }
        }
    }

    std::ostringstream out;

    out << '(' << number_of_labeled_pixels << ", " << number_of_correctly_labeled_pixels << ")\n";

    std::fstream out_file("ergebnisse.txt", std::ios::out);
    out_file << out.str();
}



// exportierte Funktionen, wenn man das Programm als Bibliothek benutzen will

extern "C" {
#ifdef _WIN32
    __declspec(dllexport)
#endif
void training(unsigned int number_of_training_images, const char** training_images, const char** label_images, const char* target_json_file, unsigned int forest_size, unsigned int max_tree_depth, unsigned int testobject_tries, unsigned int window_radius, unsigned int number_of_threads)
{
    install_signal_handler();

    FOREST_SIZE = forest_size;
    TESTOBJECT_TRIES = testobject_tries;
    MAX_TREE_DEPTH = max_tree_depth;
    WINDOW_RADIUS = window_radius;
    WINDOW_SIZE = 2*WINDOW_RADIUS + 1;

#ifdef _OPENMP
    if(number_of_threads >= 1) {
        omp_set_num_threads(number_of_threads);
    }
#endif

    // die String-Listen von char** nach vector<string> umwandeln
    std::vector<std::string> ti(training_images, training_images + number_of_training_images);
    std::vector<std::string> li(label_images, label_images + number_of_training_images);

    Forest<PixelDifferenceTest> forest = Forest<PixelDifferenceTest>::train(ti, li);

    forest.write_to_file(target_json_file);
}


#ifdef _WIN32
    __declspec(dllexport)
#endif
void inference(const char* input_image_filename, const char* json_file, const char* result_filename, double edge_weight, int inference_method, const char* intermediate_result, const char* ground_truth_image)
{
    install_signal_handler();

    PAIRWISE_ENERGY = edge_weight;
    PAIRWISE_FACTOR = exp(-PAIRWISE_ENERGY);

    Forest<PixelDifferenceTest> forest = Forest<PixelDifferenceTest>::load_from_file(json_file);

    CImg<unsigned char> input_image;
    try{
        input_image.assign(input_image_filename);
    } catch (...) {
        std::cerr << "Fehler: " << input_image_filename << " konnte nicht gelesen werden" << std::endl;
        std::exit(1);
    }
    CImg<unsigned char>* result = (inference_method == 0 ?
            forest.inference_maxflow(input_image, intermediate_result) :
            forest.inference_gibbs(input_image, intermediate_result));


    if(ground_truth_image != NULL) {
        print_result_statistics(ground_truth_image, result);
    }

    if(result_filename != NULL) {
        result->save(result_filename);
    }

    delete result;
}
}



// Wenn man das Programm als Binary baut

int main(int argc, char const *argv[])
{
    // argv in vector<string> umwandeln
    std::vector<std::string> param_vector(argv, argv+argc);


    std::string usage = "Beispiel:\n\n    Training: " + param_vector[0] + " training  -i trainingsbild1.png trainingsbild2.png  -l labels1.png labels2.png  -f forest.json  -d 8  -p 300  -t 10  -w 6\n\n    Inferenz: " + param_vector[0] + " inferenz  -i karte.png  -f forest.json  -l ausgabe.png  -e 10  -m maxflow\n";

    cimg_usage(usage.c_str());

    bool do_training = cimg_option("training", false, "Training");
    bool do_inference = cimg_option("inferenz", false, "Inferenz");
    const char* input_image_filename = cimg_option("-i", "karte.png", "Eingabebild für das Training bzw. Inferenz");
    const char* forest_file = cimg_option("-f", "forest.json", "Ausgabe- bzw. Eingabedatei mit dem Random Forest");
    const char* label_image_filename = cimg_option("-l", "karte_labels.png", "Eingabe- bzw. Ausgabebild mit Labels");
    unsigned short max_tree_depth = cimg_option("-d", 8, "Tiefe der Bäume (beim Training)");
    unsigned int testobject_tries = cimg_option("-p", 200, "Anzahl der Versuche für die Testknoten (beim Training)");
    unsigned short forest_size = cimg_option("-t", 20, "Anzahl der Bäume im Wald (beim Training)");
    unsigned char window_radius = cimg_option("-w", 4, "Radius der Fensterchen (beim Training)");
    unsigned int number_of_threads = cimg_option("-o", 1, "Anzahl der Threads (beim Training)");
    double pairwise_energy = cimg_option("-e", 10.0, "Konstantes Kantengewicht (bei der Inferenz)");
    std::string inference_method = cimg_option("-m", "maxflow", "Inferenzmethode. Entweder 'maxflow' oder 'gibbs'");

    if(cimg_option("-h", false, 0) || cimg_option("--help", false, 0)) {
        std::exit(0);
    }

    // der Benutzer muss entweder training oder inferenz angeben
    if(do_training == do_inference) {
        std::cerr << param_vector[0] << " -h für Hinweise zur Benutzung" << std::endl;
        std::exit(1);
    }


    if(do_training) {

        // da hinter -i und -l mehrere Bilder kommen können müssen wir hier
        // selbst sehen, wo die in argv sind
        int i = 1;
        for(; i < argc; ++i) {
            if(!strcmp(argv[i], "-i")) {
                break;
            }
        }
        ++i;
        int training_images_index_from = i;
        for(; i < argc; ++i) {
            if(!strncmp(argv[i], "-", 1)) {
                break;
            }
        }
        int training_images_index_to = i-1;


        i = 1;
        for(; i < argc; ++i) {
            if(!strcmp(argv[i], "-l")) {
                break;
            }
        }
        ++i;
        int label_images_index_from = i;
        for(; i < argc; ++i) {
            if(!strncmp(argv[i], "-", 1)) {
                break;
            }
        }
        int label_images_index_to = i-1;

        int number_of_training_images = training_images_index_to - training_images_index_from + 1;
        int number_of_label_images = label_images_index_to - label_images_index_from + 1;

        if(number_of_training_images != number_of_label_images) {
            std::cerr << "Fehler: Ungleiche Anzahl von Trainings- und Labelbildern" << std::endl;
            std::exit(1);
        }

        training(number_of_training_images, &argv[training_images_index_from], &argv[label_images_index_from], forest_file, forest_size, max_tree_depth, testobject_tries, window_radius, number_of_threads);

    } else {

        inference(input_image_filename, forest_file, label_image_filename, pairwise_energy, (inference_method == "maxflow" ? 0 : 1), NULL, NULL);

    }

    return 0;
}

