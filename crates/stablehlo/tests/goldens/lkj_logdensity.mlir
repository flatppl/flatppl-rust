module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<3x3xf32>) -> tensor<f32> {
    %0 = stablehlo.cholesky %arg1, lower = true : tensor<3x3xf32>
    %1 = stablehlo.iota dim = 0 : tensor<3x3xf32>
    %2 = stablehlo.iota dim = 1 : tensor<3x3xf32>
    %3 = stablehlo.compare EQ, %1, %2 : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xi1>
    %4 = stablehlo.constant dense<0.0> : tensor<3x3xf32>
    %5 = stablehlo.select %3, %0, %4 : (tensor<3x3xi1>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    %6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%5 init: %6) applies stablehlo.add across dimensions = [1] : (tensor<3x3xf32>, tensor<f32>) -> tensor<3xf32>
    %8 = stablehlo.log %7 : tensor<3xf32>
    %9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.reduce(%8 init: %9) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %11 = stablehlo.constant dense<2.0> : tensor<f32>
    %12 = stablehlo.multiply %11, %10 : tensor<f32>
    %13 = stablehlo.constant dense<1.0> : tensor<f32>
    %14 = stablehlo.subtract %arg0, %13 : tensor<f32>
    %15 = stablehlo.multiply %14, %12 : tensor<f32>
    %16 = stablehlo.constant dense<2.0> : tensor<f32>
    %17 = stablehlo.multiply %16, %arg0 : tensor<f32>
    %18 = stablehlo.constant dense<2.0> : tensor<f32>
    %19 = stablehlo.constant dense<0.0> : tensor<f32>
    %20 = stablehlo.add %17, %19 : tensor<f32>
    %21 = stablehlo.multiply %20, %18 : tensor<f32>
    %22 = stablehlo.constant dense<0.5> : tensor<f32>
    %23 = stablehlo.add %arg0, %22 : tensor<f32>
    %24 = stablehlo.multiply %16, %23 : tensor<f32>
    %25 = chlo.lgamma %23 : tensor<f32> -> tensor<f32>
    %26 = stablehlo.multiply %16, %25 : tensor<f32>
    %27 = chlo.lgamma %24 : tensor<f32> -> tensor<f32>
    %28 = stablehlo.subtract %26, %27 : tensor<f32>
    %29 = stablehlo.multiply %18, %28 : tensor<f32>
    %30 = stablehlo.constant dense<1.0> : tensor<f32>
    %31 = stablehlo.constant dense<-1.0> : tensor<f32>
    %32 = stablehlo.add %17, %31 : tensor<f32>
    %33 = stablehlo.multiply %32, %30 : tensor<f32>
    %34 = stablehlo.add %21, %33 : tensor<f32>
    %35 = stablehlo.constant dense<0.0> : tensor<f32>
    %36 = stablehlo.add %arg0, %35 : tensor<f32>
    %37 = stablehlo.multiply %16, %36 : tensor<f32>
    %38 = chlo.lgamma %36 : tensor<f32> -> tensor<f32>
    %39 = stablehlo.multiply %16, %38 : tensor<f32>
    %40 = chlo.lgamma %37 : tensor<f32> -> tensor<f32>
    %41 = stablehlo.subtract %39, %40 : tensor<f32>
    %42 = stablehlo.multiply %30, %41 : tensor<f32>
    %43 = stablehlo.add %29, %42 : tensor<f32>
    %44 = stablehlo.constant dense<0.6931471805599453> : tensor<f32>
    %45 = stablehlo.multiply %34, %44 : tensor<f32>
    %46 = stablehlo.add %45, %43 : tensor<f32>
    %47 = stablehlo.negate %46 : tensor<f32>
    %48 = stablehlo.add %15, %47 : tensor<f32>
    return %48 : tensor<f32>
  }
}
