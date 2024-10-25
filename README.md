<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/nicholicaron/chop">
    <img src="images/peutinger-table-map-1619.jpg" alt="Tabula Peutingeriana, a first century abstract depiction of roads as a network or graph" width="960" height="480">
  </a>

<h3 align="center">CHOP</h3>

  <p align="center">
CHOP (Combinatorial Heuristic Optimization Powerhouse) is a research project in which we explore ways to solve Combinatorial Optimization problems faster by using Deep Reinforcement Learning to learn better heuristics in Mixed-Integer Linear Program solvers.  
<br />
    <a href="https://github.com/nicholicaron/chop"><strong>Explore the docs (coming soon)»</strong></a>
    <br />
    <br />
    <a href="https://github.com/nicholicaron/chop">View Demo (coming soon)</a>
    ·
    <a href="https://github.com/nicholicaron/chop/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/nicholicaron/chop/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

CHOP (Combinatorial Heuristic Optimization Powerhouse) is a research project in which we explore ways to solve Combinatorial Optimization problems faster by using Graph Neural Networks to learn better heuristics in Mixed-Integer Linear Program solvers.



<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Python][python]][python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Dependencies

1. Python version 3.7 or greater
2. NumPy
3. Numba
4. NetworkX
5. Matplotlib
6. PyTorch
7. PyTorch Geometric

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/nicholicaron/chop.git
   ```
2. Install the dependencies
   ```sh
   pip install numpy numba networkx matplotlib torch torch_geometric
   ```

<!-- USAGE EXAMPLES -->
## Usage

To run the ILP solver with branch and bound:

```sh
python src/branch_and_bound.py [--visualize]
```

The `--visualize` flag is optional and will generate and save plots of the branch and bound tree.

For more detailed usage:

```python
from src.branch_and_bound import ILPSolver

# Initialize the solver
solver = ILPSolver()

# Define your problem
c = np.array([...])  # Objective coefficients
A_ub = np.array([...])  # Inequality constraint coefficients
b_ub = np.array([...])  # Inequality constraint right-hand sides
A_eq = np.array([...])  # Equality constraint coefficients (optional)
b_eq = np.array([...])  # Equality constraint right-hand sides (optional)

# Solve the problem
solution, value, num_nodes, optimal_node = solver.solve(c, A_ub, b_ub, A_eq, b_eq, problem_name="My Problem", visualize=True)

print(f"Optimal solution: {solution}")
print(f"Optimal value: {value}")
print(f"Number of nodes explored: {num_nodes}")
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

### 1. Implement Basic Linear Programming Solver:
  - [x] Implement core Simplex algorithm components:
    + [x] Initialize solution
    + [x] Choose entering variable
    + [x] Choose leaving variable
    + [x] Perform pivot operation
    + [x] Implement solution extraction and result reporting
### 2. Enhance Solver Capabilities:
  - [ ] Add support for minimization problems
### 3. Improve Numerical Stability:
  - [ ] Implement techniques to handle degeneracy
  - [ ] Add safeguards against cycling
  - [ ] Incorporate techniques for improving numerical precision
### 4. Extend Solver Functionality:
  - [x] Implement support for integer programming (branch and bound)
    + [ ] Implement more sophisticated branching variable selection strategies.
    + [ ] Add cutting plane techniques to tighten the LP relaxations.
  - [ ] Add functionality for mixed integer linear programming
### 5. Advanced Features and Research:
  - [ ] Explore and implement cutting-edge LP solving techniques
  - [ ] Investigate machine learning approaches for improving solver performance


See the [open issues](https://github.com/nicholicaron/chop/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
<!-- ## License -->

<!-- Distributed under the MIT License. See `LICENSE.txt` for more information. -->

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->





<!-- CONTACT -->
## Contact

Nicholi Caron - nmooreca@students.kennesaw.edu

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
* I want to give a huge thank you to [Dr. Misha Lavrov](https://misha.fish/) for supervising this research project.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/nicholicaron/chop.svg?style=for-the-badge
[contributors-url]: https://github.com/nicholicaron/chop/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/nicholicaron/chop.svg?style=for-the-badge
[forks-url]: https://github.com/nicholicaron/chop/network/members
[stars-shield]: https://img.shields.io/github/stars/nicholicaron/chop.svg?style=for-the-badge
[stars-url]: https://github.com/nicholicaron/chop/stargazers
[issues-shield]: https://img.shields.io/github/issues/nicholicaron/chop.svg?style=for-the-badge
[issues-url]: https://github.com/nicholicaron/chop/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge 
[licnse-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/nicholicaron
[product-screenshot]: images/screenshot.png
[python]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
