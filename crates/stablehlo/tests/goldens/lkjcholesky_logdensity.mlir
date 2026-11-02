module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<3x3xf32>) -> (tensor<f32>, tensor<i32>) {
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.compare LE, %arg0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.constant dense<9> : tensor<i32>
    %5 = stablehlo.select %2, %4, %3 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %6 = stablehlo.iota dim = 0 : tensor<3x3xf32>
    %7 = stablehlo.iota dim = 1 : tensor<3x3xf32>
    %8 = stablehlo.compare EQ, %6, %7 : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xi1>
    %9 = stablehlo.constant dense<0.0> : tensor<3x3xf32>
    %10 = stablehlo.select %8, %arg1, %9 : (tensor<3x3xi1>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    %11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%10 init: %11) applies stablehlo.add across dimensions = [1] : (tensor<3x3xf32>, tensor<f32>) -> tensor<3xf32>
    %13 = stablehlo.constant dense<2.0> : tensor<f32>
    %14 = stablehlo.multiply %13, %arg0 : tensor<f32>
    %15 = stablehlo.slice %12 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %16 = stablehlo.reshape %15 : (tensor<1xf32>) -> tensor<f32>
    %17 = stablehlo.log %16 : tensor<f32>
    %18 = stablehlo.constant dense<-1.0> : tensor<f32>
    %19 = stablehlo.add %14, %18 : tensor<f32>
    %20 = stablehlo.multiply %19, %17 : tensor<f32>
    %21 = stablehlo.slice %12 [2:3] : (tensor<3xf32>) -> tensor<1xf32>
    %22 = stablehlo.reshape %21 : (tensor<1xf32>) -> tensor<f32>
    %23 = stablehlo.log %22 : tensor<f32>
    %24 = stablehlo.constant dense<-2.0> : tensor<f32>
    %25 = stablehlo.add %14, %24 : tensor<f32>
    %26 = stablehlo.multiply %25, %23 : tensor<f32>
    %27 = stablehlo.add %20, %26 : tensor<f32>
    %28 = stablehlo.add %14, %1 : tensor<f32>
    %29 = stablehlo.multiply %28, %13 : tensor<f32>
    %30 = stablehlo.constant dense<0.5> : tensor<f32>
    %31 = stablehlo.add %arg0, %30 : tensor<f32>
    %32 = stablehlo.multiply %13, %31 : tensor<f32>
    %33 = chlo.lgamma %31 : tensor<f32> -> tensor<f32>
    %34 = stablehlo.multiply %13, %33 : tensor<f32>
    %35 = chlo.lgamma %32 : tensor<f32> -> tensor<f32>
    %36 = stablehlo.subtract %34, %35 : tensor<f32>
    %37 = stablehlo.multiply %13, %36 : tensor<f32>
    %38 = stablehlo.constant dense<1.0> : tensor<f32>
    %39 = stablehlo.multiply %19, %38 : tensor<f32>
    %40 = stablehlo.add %29, %39 : tensor<f32>
    %41 = stablehlo.add %arg0, %1 : tensor<f32>
    %42 = stablehlo.multiply %13, %41 : tensor<f32>
    %43 = chlo.lgamma %41 : tensor<f32> -> tensor<f32>
    %44 = stablehlo.multiply %13, %43 : tensor<f32>
    %45 = chlo.lgamma %42 : tensor<f32> -> tensor<f32>
    %46 = stablehlo.subtract %44, %45 : tensor<f32>
    %47 = stablehlo.multiply %38, %46 : tensor<f32>
    %48 = stablehlo.add %37, %47 : tensor<f32>
    %49 = stablehlo.constant dense<0.6931471805599453> : tensor<f32>
    %50 = stablehlo.multiply %40, %49 : tensor<f32>
    %51 = stablehlo.add %50, %48 : tensor<f32>
    %52 = stablehlo.negate %51 : tensor<f32>
    %53 = stablehlo.add %27, %52 : tensor<f32>
    return %53, %5 : tensor<f32>, tensor<i32>
  }
}
