# GPU-CT-Unmatched-back-projectors

This python toolbox provides methods to solve massive scale X-ray computerized tomography. The methods provided includes Hybrid AB-GMRES, Hybrid BA-GMRES, AB-LSQR, BA-LSQR, all of wich use unmatched projectors and are capable of solving normal and unnormal equations that come from CT problems. Portions of this project are based on code from [1], licensed under the MIT License.

The Hybrid ABBA-GMRES methods work by incorporating Tikhonav regularization into ABBA-GMRES. They support the following features:
- Dense, sparse, or abstract matrices
- Using restarting to improve efficency
- Different methods to choose regularization parameter for Tikhonav
- Multiple automatic stopping criterion
  
The ABBA-LSQR methods are iterative methods based on LSQR that can handle unmatched back projectors. They support the following features:
- Dense, sparse, or abstract matrices
- Multiple automatic stopping criterion

## Package Requirements
- numpy
- Astra
- GPUtil
- trips-py
- scipy

## Citations
[1] Maria Knudsen. ABBA-GMRES Toolbox. https://github.com/maria120123/ABBA-GMRES. Accessed: 2024-
04-7.
